# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonny Reckless
"""The shared Reckless NNUE architecture and its `RNNUE1` net container — the Python side.

One network shape and one file format for every chess engine in the organisation (`pawnstar`,
`pawnstar-rs`, `pawnstar-go` and Zenith): a net trained once loads in all of them, so developing a new
evaluation function is done once rather than once per engine. The shape is Zenith's, because its two
additions over the others' — material output buckets (+14.7 Elo) and horizontal king mirroring
(+14.3 Elo) — were measured and SPRT-gated here. See the `reckless-nnue` skill for the specification.

    (768 x 8 king buckets, horizontally mirrored) -> 1024 x2 perspectives -> 8 material heads -> 1

This module is **stdlib only**, on purpose: it is the one definition of the architecture's numbers and of
the container, and CI's synthetic-net generator (`tools/make_test_net.py`) must be able to import it on a
machine with no trainer environment. `features.py` takes its constants from here and owns everything
above them — the feature-index convention, the quantisation arithmetic and the integer forward pass.

The container is self-describing where the retired `ZNNUE5`/`ZNNUE4`/`ZNNUE3` formats were not. Those
carried an 8-byte magic and nothing else, so a net for a different architecture was told apart only by
its file size, and two different king-bucket maps of the same cardinality were indistinguishable — which
loaded cleanly and evaluated from the wrong weight bank. Every architecture field here is checked by the
loading engine against its own compile-time constants, the map is checked by digest, and three CRCs make
a corrupt or truncated file refuse rather than play.
"""

import struct

# --- the architecture -----------------------------------------------------------------------------
BASE_FEATURES = 768       # per-king-bucket feature block: 2 colors x 6 piece types x 64 squares
NUM_KING_BUCKETS = 8      # king-input buckets: 4 files (a-d, after mirroring) x 2 board-halves
INPUT_FEATURES = NUM_KING_BUCKETS * BASE_FEATURES  # 6144 feature-transformer rows
HIDDEN_SIZE = 1024        # accumulator width per perspective
NUM_OUTPUT_BUCKETS = 8    # material output heads, selected by (piece_count - 2) // 4
QUANT_ACCUMULATOR = 255   # QA: feature-transformer weight/accumulator scale
QUANT_OUTPUT = 64         # QB: output-weight scale
EVALUATION_SCALE = 400    # logit -> centipawn scale (must equal EVAL_SCALE in the trainer loss)

# Horizontal king mirroring: a perspective whose own king sits on files e-h is flipped (square ^ 7) onto
# files a-d before indexing, so every bucket trains on both wings — twice the effective data per bucket at
# the same file size. Always on in Zenith; it is a named constant because the container's format version
# and its mirroring flag are both DERIVED from it rather than written down twice.
IS_KING_MIRRORING_ENABLED = True


def _build_king_bucket_map():
    """The 64-entry king-bucket map, one entry per perspective-relative king square.

    Mirrored: `half * 4 + min(file, 7 - file)` — per *file*, because mirroring has already folded e-h onto
    d-a. Unmirrored (the retired scheme, and the other engines' pre-port shape): `half * 4 + file // 2`,
    four file-*pairs* by board-half. The two are different architectures, not one architecture with a flag:
    turning mirroring on changes what a bucket *means*, which is why the format version moves with it.
    """
    entries = bytearray(64)
    for square in range(64):
        file = square & 7
        half = (square >> 3) // 4
        folded = min(file, 7 - file) if IS_KING_MIRRORING_ENABLED else file // 2
        entries[square] = half * 4 + folded
    return bytes(entries)


KING_BUCKET_MAP = _build_king_bucket_map()

# --- CRC-64/XZ ------------------------------------------------------------------------------------
# Also called CRC-64/GO-ECMA: polynomial 0x42F0E1EBA9EA3693, reflected (table constant
# 0xC96C5795D7870F42), init and final XOR all ones, reflected in and out.
CRC64_REFLECTED_POLYNOMIAL = 0xC96C5795D7870F42
CRC64_INITIAL_AND_FINAL_XOR = 0xFFFFFFFFFFFFFFFF
CRC64_CHECK_VALUE = 0x995DC9BBDF1939FA  # the checksum of the nine ASCII bytes "123456789"

_CRC64_TABLE = []
for _index in range(256):
    _remainder = _index
    for _bit in range(8):
        _remainder = (_remainder >> 1) ^ (CRC64_REFLECTED_POLYNOMIAL if _remainder & 1 else 0)
    _CRC64_TABLE.append(_remainder)


def crc64_xz(data):
    """CRC-64/XZ of `data`, init and final XOR applied. Returns the 64-bit checksum."""
    running = CRC64_INITIAL_AND_FINAL_XOR
    for byte in data:
        running = _CRC64_TABLE[(running ^ byte) & 0xFF] ^ (running >> 8)
    return running ^ CRC64_INITIAL_AND_FINAL_XOR


# The published check value, asserted at import. Every Reckless engine asserts this same one: a CRC with
# the wrong reflection or a wrong table constant still returns a plausible 64-bit number, so without this
# the first symptom would be each engine refusing every net the others stamped.
assert crc64_xz(b"123456789") == CRC64_CHECK_VALUE, "CRC-64/XZ is wrong in this module"
assert crc64_xz(b"") == 0, "CRC-64/XZ of no bytes must be zero"

# --- the container --------------------------------------------------------------------------------
MAGIC = b"RNNUE1\0\0"
HEADER_BYTES = 64
FLAG_KING_MIRRORING = 0x01  # flags bit 0
ACTIVATION_SCRELU = 0       # activation 0: clamp(x, 0, QA), which the output dot then squares

# 2 = the mirrored map with material output heads; 1 = the unmirrored file-pair map with one head.
FORMAT_VERSION = 2 if IS_KING_MIRRORING_ENABLED else 1
FLAGS = FLAG_KING_MIRRORING if IS_KING_MIRRORING_ENABLED else 0

# Exact size of the quantised payload: three int16 sections then one int32 per output bucket, tightly
# packed and little-endian, with nothing after it. 12,617,760 bytes at this shape.
PAYLOAD_BYTES = (
    (INPUT_FEATURES * HIDDEN_SIZE + HIDDEN_SIZE + NUM_OUTPUT_BUCKETS * 2 * HIDDEN_SIZE) * 2
    + NUM_OUTPUT_BUCKETS * 4
)

# magic, format_version, header_bytes, features_per_bucket, king_buckets, hidden_size, output_buckets,
# flags, activation, qa, qb, eval_scale, 4 reserved zero bytes, map CRC, payload length, payload CRC.
_HEADER_THROUGH_PAYLOAD_CRC = "<8sHHHHHHBBHHH4xQQQ"
assert struct.calcsize(_HEADER_THROUGH_PAYLOAD_CRC) == HEADER_BYTES - 8


def header(payload):
    """The 64-byte RNNUE1 header for `payload`, written from this module's own constants.

    Nothing here is passed in but the weights: the architecture fields, the map digest and the payload
    length all come from the constants above, so a payload of the wrong shape gets a header that every
    engine refuses instead of a header that describes it. Raises ValueError on a wrong-length payload,
    which is the same mistake caught one step earlier.
    """
    if len(payload) != PAYLOAD_BYTES:
        raise ValueError(f"payload is {len(payload)} bytes; this architecture's payload is {PAYLOAD_BYTES}")
    through_payload_crc = struct.pack(
        _HEADER_THROUGH_PAYLOAD_CRC, MAGIC, FORMAT_VERSION, HEADER_BYTES, BASE_FEATURES, NUM_KING_BUCKETS,
        HIDDEN_SIZE, NUM_OUTPUT_BUCKETS, FLAGS, ACTIVATION_SCRELU, QUANT_ACCUMULATOR, QUANT_OUTPUT,
        EVALUATION_SCALE, crc64_xz(KING_BUCKET_MAP), PAYLOAD_BYTES, crc64_xz(payload))
    return through_payload_crc + struct.pack("<Q", crc64_xz(through_payload_crc))


def write_net(path, payload):
    """Write `payload` to `path` in the RNNUE1 container. Returns the number of bytes written."""
    image = header(payload) + payload
    with open(path, "wb") as out:
        out.write(image)
    return len(image)


def read_payload(path):
    """Read a net's weight payload back, checking the magic, the length and both CRCs.

    This is a reader for the Python tools (the verification gate, a re-stamp); the engine's own loader is
    the one that matters and checks every architecture field besides. Raises SystemExit on a refusal.
    """
    with open(path, "rb") as source:
        image = source.read()
    if len(image) < HEADER_BYTES or image[:8] != MAGIC:
        raise SystemExit(f"{path}: not a net in the RNNUE1 container")
    if crc64_xz(image[:HEADER_BYTES - 8]) != struct.unpack_from("<Q", image, HEADER_BYTES - 8)[0]:
        raise SystemExit(f"{path}: its header is corrupt")
    payload = image[HEADER_BYTES:]
    expected_length, expected_crc = struct.unpack_from("<QQ", image, 40)
    if len(payload) != expected_length:
        raise SystemExit(f"{path}: it carries {len(payload)} payload bytes, not the {expected_length} it claims")
    if crc64_xz(payload) != expected_crc:
        raise SystemExit(f"{path}: its payload is corrupt")
    return payload
