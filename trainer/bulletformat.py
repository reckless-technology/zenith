# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonny Reckless
"""Vectorised reader and featuriser for `bulletformat` training records.

The trainer's original front end reads `fen;score;wdl` text and featurises it one position at a time in
Python (`train._featurise_lines`), which is fine once per shard into an `.npz` cache and far too slow on a
training critical path. A `bulletformat` record is already a packed bitboard rather than text, so there is
no expensive parse to amortise and the cache stops being necessary: this module expands records straight
into the padded index arrays the model consumes, with numpy doing every position at once.

The record is `bulletformat::ChessBoard` (crate 1.8.0, `src/chess.rs`), 32 bytes, `#[repr(C)]`, no header:

    offset  size  field
         0     8  occ        occupancy bitboard, little-endian u64
         8    16  pcs        one 4-bit piece per occupied square, LOW nibble of each byte first
        24     2  score      centipawns, i16
        26     1  result     0 = loss, 1 = draw, 2 = win
        27     1  ksq        side-to-move's own king square
        28     1  opp_ksq    opponent's king square, ALREADY XOR 56
        29     3  extra

Two properties of the format do most of the work here, and both are easy to get wrong.

**The record is stored side-to-move relative, not White relative.** `ChessBoard::from_raw` and its
`FromStr` both flip a black-to-move position before storing it: every square `^ 56`, every piece's colour
bit inverted, `score = -score` and `result = 2 - result`. So the stored board always has the side to move
playing up the board, the nibble's colour bit is 0 for the side to move's OWN pieces and 1 for the
opponent's, and `score` and `result` are already from the side to move's point of view. The crate's doc
comment "Score is White relative" describes `from_raw`'s *argument*, before that flip, not the stored
field -- a distinction worth keeping straight, because reading the stored score as White relative
correlates an exported net against its own training labels at about 0.14 instead of about 0.78.

**The nibbles are in ascending square order, not piece order.** `BoardIter` walks `occ` by
`trailing_zeros`, so the n-th nibble belongs to the n-th lowest set bit of the occupancy. Each row has a
different piece count, which is the one genuinely awkward step to vectorise: `np.nonzero` gives every
active square in row-major (so per-row ascending) order, and subtracting a `cumsum` row offset turns that
into each piece's nibble slot.

`opp_ksq` being pre-XORed is a convenience rather than a wrinkle: `features.feature_index` wants each
perspective's king square in that perspective's own frame, which for the opponent is exactly `square ^ 56`.
"""

import os

import numpy as np

from features import MAX_ACTIVE_FEATURES, PADDING_INDEX
from rnnue1 import BASE_FEATURES, KING_BUCKET_MAP

RECORD_BYTES = 32
PIECES_PER_COLOUR_BLOCK = 384  # one perspective's own-pieces block: 6 piece types x 64 squares

# The record's own field layout, used for the scalar fields; the occupancy bitboard is read through the
# plain uint8 view instead, because np.unpackbits wants bytes.
RECORD_DTYPE = np.dtype([
    ("occupancy", "<u8"),
    ("pieces", "u1", (16,)),
    ("score", "<i2"),
    ("result", "u1"),
    ("own_king_square", "u1"),
    ("opponent_king_square", "u1"),
    ("extra", "u1", (3,)),
])
assert RECORD_DTYPE.itemsize == RECORD_BYTES, "the bulletformat record is 32 bytes"

_KING_BUCKET_MAP_ARRAY = np.frombuffer(KING_BUCKET_MAP, dtype=np.uint8).astype(np.int16)
_SLOT_NUMBERS = np.arange(MAX_ACTIVE_FEATURES, dtype=np.int16)

# nibble -> (piece_type + 6 * relative_colour) * 64, one table per perspective. A nibble is
# `colour << 3 | piece_type`, with colour 0 meaning "belongs to the side to move", so the colour block and
# the piece-type block of the feature index are a function of the nibble alone. Reading them out of a
# 16-entry table costs one cache-resident gather in place of two shifts, two multiplies and two widening
# casts over every piece in the block, which measured as the second-largest cost in the whole featurisation.
_OWN_PIECE_OFFSETS = np.zeros(16, dtype=np.int16)
_OPPONENT_PIECE_OFFSETS = np.zeros(16, dtype=np.int16)
for _nibble in range(16):
    _colour, _piece_type = _nibble >> 3, _nibble & 7
    if _piece_type < 6:  # 6 and 7 are not pieces; those slots are never read
        _OWN_PIECE_OFFSETS[_nibble] = (_piece_type + 6 * _colour) * 64
        _OPPONENT_PIECE_OFFSETS[_nibble] = (_piece_type + 6 * (1 - _colour)) * 64

PIECE_TYPE_CHARACTERS = "PNBRQK"


def record_count_of(path):
    """How many whole records a file holds."""
    size = os.path.getsize(path)
    if size % RECORD_BYTES:
        raise ValueError(f"{path}: {size} bytes is not a whole number of {RECORD_BYTES}-byte records")
    return size // RECORD_BYTES


def read_records(path, count=None, first_record=0):
    """Memory-map `path` and return `count` records from `first_record` as a structured array.

    The return value is a view into the mapping, so nothing is read until it is touched and a slice of a
    123 GB file costs what the slice costs. Pass `count=None` for every record from `first_record` on.
    """
    available = record_count_of(path) - first_record
    if available <= 0:
        raise ValueError(f"{path}: no records at or after record {first_record}")
    if count is None:
        count = available
    count = min(count, available)
    mapped = np.memmap(path, dtype=np.uint8, mode="r",
                       offset=first_record * RECORD_BYTES, shape=(count * RECORD_BYTES,))
    return mapped.view(RECORD_DTYPE)


def featurise_records(records):
    """Expand a block of records into the padded index arrays the model consumes.

    Returns `(own_indices, opponent_indices, scores, results)`:
      * `own_indices`, `opponent_indices` -- `(count, MAX_ACTIVE_FEATURES)` int16, padded with
        `PADDING_INDEX`, each row's active entries in ascending square order
      * `scores` -- float32 centipawns, side-to-move relative
      * `results` -- float32 in {0.0, 0.5, 1.0}, side-to-move relative

    Every step is whole-array. Rows carrying more than `MAX_ACTIVE_FEATURES` pieces cannot occur in a legal
    position and are rejected rather than silently truncated.

    The shape of the code follows a measurement rather than taste. Only the square of each piece needs the
    variable-length treatment, because a piece's slot in the nibble array is already its slot in the output
    array; so the squares are compacted once and everything after that is dense `(count, 32)` arithmetic
    with the per-row quantities broadcast, which avoids gathering the king bucket and the mirror once per
    piece. Within that, the piece counts come from `np.bitwise_count` on the bitboard rather than from
    summing the unpacked bits (45 ms to 1 ms per million records), the compaction uses `np.flatnonzero`
    over the ravelled bits rather than `np.nonzero` over the two-dimensional array (405 ms to 235 ms, and
    one index array instead of two), and the colour and piece-type terms come from the two 16-entry tables
    above. Together those took one million records from 0.67M to 1.13M positions/sec on one core, with
    byte-identical output; what remains is bounded by memory bandwidth over the two 67 MB output arrays and
    by the compaction's indirection, and did not move further for any arrangement tried.
    """
    records = np.ascontiguousarray(records)
    count = records.shape[0]
    bytes_view = records.view(np.uint8).reshape(count, RECORD_BYTES)

    occupancy_bytes = np.ascontiguousarray(bytes_view[:, 0:8])
    piece_counts = np.bitwise_count(occupancy_bytes.view(np.uint64).reshape(count)).astype(np.int16)
    if piece_counts.size and piece_counts.max() > MAX_ACTIVE_FEATURES:
        raise ValueError(f"a record claims {int(piece_counts.max())} pieces; at most {MAX_ACTIVE_FEATURES} fit")

    # Occupied squares. bitorder="little" makes column j of the unpacked array square j: byte 0 carries
    # squares 0-7 with bit 0 the lowest, which is what a little-endian u64 bitboard means.
    occupancy_bits = np.unpackbits(occupancy_bytes, axis=1, bitorder="little")

    # One flat offset per occupied square, rows ascending and squares ascending within a row -- which is
    # the order the nibbles were written in. The row and the square fall out of it by a shift and a mask.
    occupied = np.flatnonzero(occupancy_bits.ravel())
    active_rows = occupied >> 6
    active_squares = (occupied & 63).astype(np.int16)

    # Each piece's slot within its own row: its flat position less the row's start. This is the one step
    # the variable piece count makes awkward, and it is two whole-array operations.
    row_starts = np.zeros(count, dtype=np.int64)
    if count > 1:
        np.cumsum(piece_counts[:-1], out=row_starts[1:])
    active_slots = np.arange(occupied.size, dtype=np.int64) - row_starts[active_rows]

    squares = np.zeros((count, MAX_ACTIVE_FEATURES), dtype=np.int16)
    squares[active_rows, active_slots] = active_squares

    # The packed nibbles, low nibble of each byte first. Slot j here is slot j of the output.
    packed_pieces = np.ascontiguousarray(bytes_view[:, 8:24])
    nibbles = np.empty((count, MAX_ACTIVE_FEATURES), dtype=np.uint8)
    nibbles[:, 0::2] = packed_pieces & 0x0F
    nibbles[:, 1::2] = packed_pieces >> 4

    own_king_squares = bytes_view[:, 27].astype(np.int16)
    opponent_king_squares = bytes_view[:, 28].astype(np.int16)

    # Horizontal mirroring and the king bucket, per row, from each perspective's own king square.
    own_bases = (_KING_BUCKET_MAP_ARRAY[own_king_squares] * BASE_FEATURES)[:, None]
    opponent_bases = (_KING_BUCKET_MAP_ARRAY[opponent_king_squares] * BASE_FEATURES)[:, None]
    own_mirrors = np.where((own_king_squares & 7) >= 4, 7, 0).astype(np.int16)[:, None]
    opponent_mirrors = np.where((opponent_king_squares & 7) >= 4, 7, 0).astype(np.int16)[:, None]

    # features.feature_index, one expression per perspective. The side-to-move perspective reads the record
    # frame as it stands; the opponent perspective flips every square vertically and swaps which colour
    # block a piece lands in. Accumulated in place to keep the number of 67 MB temporaries down.
    own_indices = _OWN_PIECE_OFFSETS[nibbles]
    own_indices += own_bases
    own_indices += squares ^ own_mirrors
    opponent_indices = _OPPONENT_PIECE_OFFSETS[nibbles]
    opponent_indices += opponent_bases
    opponent_indices += (squares ^ 56) ^ opponent_mirrors

    inactive = _SLOT_NUMBERS[None, :] >= piece_counts[:, None]
    np.copyto(own_indices, np.int16(PADDING_INDEX), where=inactive)
    np.copyto(opponent_indices, np.int16(PADDING_INDEX), where=inactive)

    scores = np.ascontiguousarray(bytes_view[:, 24:26]).view(np.int16).reshape(count).astype(np.float32)
    results = bytes_view[:, 26].astype(np.float32) * 0.5
    return own_indices, opponent_indices, scores, results


def record_to_fen(record):
    """Reconstruct one record's position as a FEN, in the record's own side-to-move-relative frame.

    The frame is the point: the record does not say whose turn it was, because it has already flipped a
    black-to-move position onto the side to move. So the FEN this returns is always "w" to move, and
    `features.position_features` of it yields exactly the indices `featurise_records` yields for the same
    record. Castling rights and the clocks are not in the record and are written as unavailable; neither
    affects featurisation. For the equivalence check and for reading records by eye, not for training.
    """
    occupancy = int(record["occupancy"])
    packed = np.asarray(record["pieces"])
    board = {}
    slot = 0
    while occupancy:
        square = (occupancy & -occupancy).bit_length() - 1
        occupancy &= occupancy - 1
        nibble = int(packed[slot // 2] >> (4 * (slot & 1))) & 0x0F
        character = PIECE_TYPE_CHARACTERS[nibble & 0x07]
        board[square] = character if (nibble >> 3) == 0 else character.lower()
        slot += 1

    rows = []
    for rank in range(7, -1, -1):
        row = ""
        empty = 0
        for file in range(8):
            character = board.get(rank * 8 + file)
            if character is None:
                empty += 1
                continue
            if empty:
                row += str(empty)
                empty = 0
            row += character
        if empty:
            row += str(empty)
        rows.append(row)
    return "/".join(rows) + " w - - 0 1"
