"""Shared NNUE constants, FEN featurisation, and the integer forward pass.

This module is the single source of truth for the feature-index convention and the quantisation contract.
The C++ engine (src/nnue.*) must reproduce `feature_index` and `integer_eval` byte-for-byte — the
verification gate compares the engine's integer eval against `integer_eval` here on a fixed FEN set.

Architecture (Zenith v1, independent of any other engine):
    768 inputs per perspective (6 piece types x 2 colours x 64 squares, side-to-move's own pieces first)
    768 -> HIDDEN_SIZE feature transformer (shared weights for both perspectives)
    concat[own(HIDDEN), opp(HIDDEN)] -> SCReLU -> 1 output
"""

import numpy as np

INPUT_FEATURES = 768
HIDDEN_SIZE = 512
QUANT_ACCUMULATOR = 255  # QA: feature-transformer weight/accumulator scale
QUANT_OUTPUT = 64        # QB: output-weight scale
EVALUATION_SCALE = 400   # logit -> centipawn scale (must equal EVAL_SCALE in the trainer loss)

PADDING_INDEX = INPUT_FEATURES  # embedding row 768 is a forced-zero pad slot
MAX_ACTIVE_FEATURES = 32        # at most 32 pieces on the board

NNUE_MAGIC = b"ZNNUE1\0\0"  # 8-byte little-endian file magic

WHITE, BLACK = 0, 1
_PIECE_CHAR_TO_TYPE = {"p": 0, "n": 1, "b": 2, "r": 3, "q": 4, "k": 5}


def parse_board_pieces(board_field):
    """Parse the piece-placement field of a FEN into a list of (colour, piece_type, square)."""
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
            colour = WHITE if character.isupper() else BLACK
            piece_type = _PIECE_CHAR_TO_TYPE[character.lower()]
            square = rank * 8 + file
            pieces.append((colour, piece_type, square))
            file += 1
    return pieces


def feature_index(perspective, colour, piece_type, square):
    """Index of one (colour, piece_type, square) feature in the given perspective's 768-wide input.

    The side-to-move's own pieces occupy indices [0,384); the opponent's occupy [384,768). Black's
    perspective mirrors the board vertically (square ^ 56) so both sides see the board "from their side".
    """
    relative_colour = 0 if colour == perspective else 1
    relative_square = square if perspective == WHITE else (square ^ 56)
    return relative_colour * 384 + piece_type * 64 + relative_square


def position_features(fen):
    """Return (side_to_move, own_perspective_indices, opponent_perspective_indices) for a FEN."""
    fields = fen.split()
    board_field = fields[0]
    side_to_move = WHITE if fields[1] == "w" else BLACK
    own_indices = []
    opponent_indices = []
    for colour, piece_type, square in parse_board_pieces(board_field):
        own_indices.append(feature_index(side_to_move, colour, piece_type, square))
        opponent_indices.append(feature_index(1 - side_to_move, colour, piece_type, square))
    return side_to_move, own_indices, opponent_indices


def _truncated_divide(numerator, denominator):
    """Integer division truncating toward zero, matching C++ operator/ (Python // floors instead)."""
    quotient = abs(numerator) // abs(denominator)
    return quotient if (numerator < 0) == (denominator < 0) else -quotient


def integer_eval(feature_transformer_weight, feature_transformer_bias, output_weight, output_bias,
                 own_indices, opponent_indices):
    """Reference integer forward pass; MUST stay identical to nnue::evaluate in the C++ engine.

    Arguments are the quantised arrays exactly as stored in the .nnue file:
        feature_transformer_weight : int array shaped [768, HIDDEN_SIZE]
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
