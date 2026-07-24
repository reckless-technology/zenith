# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonny Reckless
"""Shared NNUE constants, FEN featurisation, and the integer forward pass.

This module is the single source of truth for the feature-index convention and the quantisation contract.
The C engine (src/nnue.*) must reproduce `feature_index` and `integer_eval` byte-for-byte — the
verification gate compares the engine's integer eval against `integer_eval` here on a fixed FEN set.

Architecture (Zenith v2, independent of any other engine):
    768 base inputs per perspective (6 piece types x 2 colors x 64 squares, own pieces first), replicated
    across NUM_KING_BUCKETS king-input buckets selected by the perspective's own king square (file-pair x
    board-half) -> HIDDEN_SIZE feature transformer -> concat[own(HIDDEN), opp(HIDDEN)] -> SCReLU -> 1 output
"""

import numpy as np

BASE_FEATURES = 768      # per-king-bucket feature block: 2 colors x 6 piece types x 64 squares
NUM_KING_BUCKETS = 8     # king-input buckets: 4 file-pairs x 2 board-halves, keyed on the perspective king
INPUT_FEATURES = NUM_KING_BUCKETS * BASE_FEATURES  # 6144 feature-transformer rows
HIDDEN_SIZE = 512
QUANT_ACCUMULATOR = 255  # QA: feature-transformer weight/accumulator scale
QUANT_OUTPUT = 64        # QB: output-weight scale
EVALUATION_SCALE = 400   # logit -> centipawn scale (must equal EVAL_SCALE in the trainer loss)

PADDING_INDEX = INPUT_FEATURES  # embedding row 6144 is a forced-zero pad slot
MAX_ACTIVE_FEATURES = 32        # at most 32 pieces on the board

NNUE_MAGIC = b"ZNNUE3\0\0"  # 8-byte little-endian file magic (v3: king-input buckets)

WHITE, BLACK = 0, 1
KING = 5
_PIECE_CHAR_TO_TYPE = {"p": 0, "n": 1, "b": 2, "r": 3, "q": 4, "k": 5}


def king_bucket(relative_king_square):
    """Map a perspective-relative king square (0..63) to one of NUM_KING_BUCKETS buckets: 4 file-pairs
    (a/b, c/d, e/f, g/h) x 2 board-halves (ranks 1-4, ranks 5-8). Must match king_bucket() in src/nnue.c."""
    file_pair = (relative_king_square & 7) // 2  # 0..3
    half = (relative_king_square >> 3) // 4      # 0..1
    return half * 4 + file_pair                  # 0..7


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
        output_weight              : int array shaped [2 * HIDDEN_SIZE]  (own half then opponent half)
        output_bias                : int scalar
    Returns the evaluation in centipawns from the side-to-move's point of view.
    """
    own_accumulator = feature_transformer_bias.astype(np.int64).copy()
    for index in own_indices:
        own_accumulator += feature_transformer_weight[index]
    opponent_accumulator = feature_transformer_bias.astype(np.int64).copy()
    for index in opponent_indices:
        opponent_accumulator += feature_transformer_weight[index]

    output_weight_own = output_weight[:HIDDEN_SIZE].astype(np.int64)
    output_weight_opponent = output_weight[HIDDEN_SIZE:].astype(np.int64)

    clamped_own = np.clip(own_accumulator, 0, QUANT_ACCUMULATOR)
    clamped_opponent = np.clip(opponent_accumulator, 0, QUANT_ACCUMULATOR)
    accumulated = int(np.sum((clamped_own * output_weight_own) * clamped_own)
                      + np.sum((clamped_opponent * output_weight_opponent) * clamped_opponent))

    accumulated = _truncated_divide(accumulated, QUANT_ACCUMULATOR)
    accumulated += int(output_bias)
    return _truncated_divide(accumulated * EVALUATION_SCALE, QUANT_ACCUMULATOR * QUANT_OUTPUT)
