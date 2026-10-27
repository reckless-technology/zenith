# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonny Reckless
"""FEN featurisation, the quantisation contract, and the integer forward pass.

This module is the single source of truth for the feature-index convention and the quantisation
arithmetic. The C engine (src/nnue.*) must reproduce `feature_index` and `integer_eval` byte-for-byte —
the verification gate compares the engine's integer eval against `integer_eval` here on a fixed FEN set.
The architecture's numbers and the on-disk container come from `rnnue1.py`, which every Reckless engine
shares; nothing about the shape is written down twice.

Architecture (the shared Reckless NNUE — one shape for every engine in the organisation):
    768 base inputs per perspective (6 piece types x 2 colors x 64 squares, own pieces first), replicated
    across NUM_KING_BUCKETS king-input buckets selected by the perspective's own king square (file x
    board-half, with files e-h MIRRORED onto a-d: the whole perspective flips horizontally, square ^ 7, so
    every bucket sees both wings' data) -> HIDDEN_SIZE feature transformer ->
    concat[own(HIDDEN), opp(HIDDEN)] -> SCReLU ->
    NUM_OUTPUT_BUCKETS output heads, one selected per position by total piece count ((count - 2) // 4).
    Material-bucketed heads let the same hidden features be weighed differently by game phase.
"""

import numpy as np

from rnnue1 import (BASE_FEATURES, EVALUATION_SCALE, HIDDEN_SIZE, INPUT_FEATURES, KING_BUCKET_MAP,
                    NUM_KING_BUCKETS, NUM_OUTPUT_BUCKETS, QUANT_ACCUMULATOR, QUANT_OUTPUT)

PADDING_INDEX = INPUT_FEATURES  # embedding row 6144 is a forced-zero pad slot
MAX_ACTIVE_FEATURES = 32        # at most 32 pieces on the board

WHITE, BLACK = 0, 1
KING = 5
_PIECE_CHAR_TO_TYPE = {"p": 0, "n": 1, "b": 2, "r": 3, "q": 4, "k": 5}


def output_bucket(piece_count):
    """Material output bucket for a position with piece_count total pieces (kings included): 2..32 pieces
    map to buckets 0..7 via (count - 2) // 4. Must match output_bucket() in src/nnue.c."""
    return (piece_count - 2) // 4


def king_mirror(relative_king_square):
    """Whether this perspective mirrors horizontally: true when its king is on files e-h. Mirrored
    perspectives XOR every square with 7, folding the king onto files a-d."""
    return (relative_king_square & 7) >= 4


def king_bucket(relative_king_square):
    """Map a perspective-relative king square (0..63) to one of NUM_KING_BUCKETS buckets: 4 files (a-d
    after mirroring) x 2 board-halves (ranks 1-4, ranks 5-8).

    Indexes `rnnue1.KING_BUCKET_MAP`, which is also what the container's `king_bucket_map_crc` digests and
    what src/nnue.c's own KING_BUCKET_MAP is copied from. The map's *contents* are part of the
    architecture, so there is deliberately one definition of it and no arithmetic restatement to drift.
    """
    return KING_BUCKET_MAP[relative_king_square]


def parse_board_pieces(board_field):
    """Parse the piece-placement field of a FEN into a list of (color, piece_type, square)."""
    pieces = []
    rank = 7  # FEN lists rank 8 first
    file = 0
    for character in board_field:
        if character == "/":
            rank -= 1
            file = 0
        elif character.isdigit():
            file += int(character)
        else:
            color = WHITE if character.isupper() else BLACK
            piece_type = _PIECE_CHAR_TO_TYPE[character.lower()]
            square = rank * 8 + file
            pieces.append((color, piece_type, square))
            file += 1
    return pieces


def feature_index(perspective, king_square, color, piece_type, square):
    """Index of one (color, piece_type, square) feature in the given perspective's input.

    Within a king bucket the side-to-move's own pieces occupy [0,384) and the opponent's [384,768); Black's
    perspective mirrors the board vertically (square ^ 56). The perspective's own king square (also mirrored
    for Black) selects the king bucket, offsetting the whole 768 block by king_bucket * 768.
    """
    relative_color = 0 if color == perspective else 1
    if perspective == WHITE:
        relative_square = square
        relative_king = king_square
    else:
        relative_square = square ^ 56
        relative_king = king_square ^ 56
    if king_mirror(relative_king):
        relative_square ^= 7  # king on e-h: the whole perspective mirrors horizontally
    return king_bucket(relative_king) * BASE_FEATURES + relative_color * 384 + piece_type * 64 + relative_square


def position_features(fen):
    """Return (side_to_move, own_perspective_indices, opponent_perspective_indices) for a FEN.

    Each perspective's features are king-bucketed on that perspective's OWN king square, so both kings are
    located first, then every piece is emitted twice (once per perspective) with the bucket offset baked in.
    """
    fields = fen.split()
    board_field = fields[0]
    side_to_move = WHITE if fields[1] == "w" else BLACK
    pieces = parse_board_pieces(board_field)

    king_square = [None, None]
    for color, piece_type, square in pieces:
        if piece_type == KING:
            king_square[color] = square
    opponent = 1 - side_to_move

    own_indices = []
    opponent_indices = []
    for color, piece_type, square in pieces:
        own_indices.append(feature_index(side_to_move, king_square[side_to_move], color, piece_type, square))
        opponent_indices.append(feature_index(opponent, king_square[opponent], color, piece_type, square))
    return side_to_move, own_indices, opponent_indices


def _truncated_divide(numerator, denominator):
    """Integer division truncating toward zero, matching C operator/ (Python // floors instead)."""
    quotient = abs(numerator) // abs(denominator)
    return quotient if (numerator < 0) == (denominator < 0) else -quotient


def integer_eval(feature_transformer_weight, feature_transformer_bias, output_weight, output_bias,
                 own_indices, opponent_indices):
    """Reference integer forward pass; MUST stay identical to nnue::evaluate in the C engine.

    Arguments are the quantised arrays exactly as stored in the .nnue file:
        feature_transformer_weight : int array shaped [INPUT_FEATURES, HIDDEN_SIZE]  (6144 king-bucketed rows)
        feature_transformer_bias   : int array shaped [HIDDEN_SIZE]
        output_weight              : int array shaped [NUM_OUTPUT_BUCKETS, 2 * HIDDEN_SIZE]
        output_bias                : int array shaped [NUM_OUTPUT_BUCKETS]
    The output head is selected by the position's piece count (= len(own_indices), every piece emits one
    feature per perspective). Returns the evaluation in centipawns from the side-to-move's point of view.
    """
    own_accumulator = feature_transformer_bias.astype(np.int64).copy()
    for index in own_indices:
        own_accumulator += feature_transformer_weight[index]
    opponent_accumulator = feature_transformer_bias.astype(np.int64).copy()
    for index in opponent_indices:
        opponent_accumulator += feature_transformer_weight[index]

    bucket = output_bucket(len(own_indices))
    bucket_weight = output_weight[bucket]
    output_weight_own = bucket_weight[:HIDDEN_SIZE].astype(np.int64)
    output_weight_opponent = bucket_weight[HIDDEN_SIZE:].astype(np.int64)

    clamped_own = np.clip(own_accumulator, 0, QUANT_ACCUMULATOR)
    clamped_opponent = np.clip(opponent_accumulator, 0, QUANT_ACCUMULATOR)
    accumulated = int(np.sum((clamped_own * output_weight_own) * clamped_own)
                      + np.sum((clamped_opponent * output_weight_opponent) * clamped_opponent))

    accumulated = _truncated_divide(accumulated, QUANT_ACCUMULATOR)
    accumulated += int(output_bias[bucket])
    return _truncated_divide(accumulated * EVALUATION_SCALE, QUANT_ACCUMULATOR * QUANT_OUTPUT)
