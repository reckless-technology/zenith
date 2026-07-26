// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief Core value types: colors, pieces, squares, and the packed 16-bit move.
 */
#pragma once
#include <stdbool.h>
#include <stdint.h>

typedef uint64_t Bitboard; ///< A 64-bit set of squares (bit i = square i, A1=0 … H8=63).

/** @brief Side to move / piece color. */
typedef enum
{
    WHITE,
    BLACK,
    NUM_COLORS = 2 ///< number of colors
} Color;

static inline Color enemy_of(Color c)
{
    return (Color)(c ^ 1);
}

/**
 * @brief The one piece type: color-independent, with NO_PIECE = 0 marking an empty square.
 *
 * The mailbox stores these directly (color lives in the Position.colors bitboards — see position_color_on),
 * and Position.pieces is indexed by them, with the NO_PIECE slot holding the occupied-squares bitboard.
 */
typedef enum
{
    NO_PIECE, ///< empty square / "no piece" sentinel (Position.pieces[NO_PIECE] = occupied squares)
    PAWN,
    KNIGHT,
    BISHOP,
    ROOK,
    QUEEN,
    KING,
    NUM_PIECES ///< array bound: NO_PIECE + the six real piece types
} Piece;

/** @brief A board square: A1 = 0 … H8 = 63 (rank = sq/8, file = sq%8), or NO_SQUARE off the board. */
typedef enum
{
    NO_SQUARE = 64 ///< off-board / "no square" sentinel
} Square;

enum
{
    NUM_SQUARES = 64 ///< number of board squares
};

static inline int rank_of(int square)
{
    return square >> 3;
}

static inline int file_of(int square)
{
    return square & 7;
}

static inline int make_square(int file, int rank)
{
    return rank * 8 + file;
}

static inline int relative_rank(Color color, int square)
{
    return color == WHITE ? rank_of(square) : 7 - rank_of(square);
}

/**
 * @brief Format a square as coordinate text.
 * @param square board square, or an off-board value (>= 64) for "-".
 * @param buf destination, needs >= 3 bytes.
 * @return @p buf, holding "-" for off-board squares else e.g. "e4" (NUL-terminated).
 */
static inline char *sq_name(int square, char *buf)
{
    if (square >= 64)
    {
        buf[0] = '-';
        buf[1] = '\0';
        return buf;
    }
    buf[0] = (char)('a' + file_of(square));
    buf[1] = (char)('1' + rank_of(square));
    buf[2] = '\0';
    return buf;
}

/** @brief Castling rights bitmask (OR of the four side/flank flags). */
enum
{
    MAY_WHITE_CASTLE_KINGSIDE  = 1, ///< white kingside
    MAY_WHITE_CASTLE_QUEENSIDE = 2, ///< white queenside
    MAY_BLACK_CASTLE_KINGSIDE  = 4, ///< black kingside
    MAY_BLACK_CASTLE_QUEENSIDE = 8, ///< black queenside
    FULL_CASTLING_RIGHTS       = 15 ///< all four rights
};

/**
 * @brief Move flags (CPW encoding): the flag nibble of a packed move (from 0..5 | to 6..11 | flag 12..15).
 *
 * Bit 3 (>= 8) marks a promotion, bit 2 (+4) marks a capture, and the low two bits carry the promotion
 * piece. Only the two promotion base flags are named: add_promotions builds the bishop/rook/queen variants
 * (and their capture forms) as base + (piece - KNIGHT), and move_promo_pt decodes them as KNIGHT + (flag & 3).
 */
enum
{
    FLAG_QUIET                    = 0,
    FLAG_PAWN_DOUBLE_PUSH         = 1,
    FLAG_CASTLE_KINGSIDE          = 2,
    FLAG_CASTLE_QUEENSIDE         = 3,
    FLAG_CAPTURE                  = 4,
    FLAG_EP                       = 5,
    FLAG_PROMOTION_KNIGHT         = 8, ///< promotion base; +1/+2/+3 -> bishop/rook/queen
    FLAG_PROMOTION_KNIGHT_CAPTURE = FLAG_PROMOTION_KNIGHT | FLAG_CAPTURE, ///< +1/+2/+3 -> bishop/rook/queen
};

/**
 * @brief A packed move: a bare uint16 (from | to<<6 | flag<<12); MOVE_NONE (0) is "no move".
 *
 * A plain integer typedef so == / != and array indexing work directly, with the accessors below for fields.
 */
typedef uint16_t Move;

#define MOVE_NONE ((Move)0) ///< the "no move" sentinel (a packed move of 0)

static inline Move move_make(int from, int to, unsigned flag)
{
    return (Move)(from | (to << 6) | (flag << 12));
}

static inline int move_from(Move move)
{
    return move & 0x3f;
}

static inline int move_to(Move move)
{
    return (move >> 6) & 0x3f;
}

static inline unsigned move_flag(Move move)
{
    return move >> 12;
}

static inline bool move_is_none(Move move)
{
    return move == 0;
}

static inline bool move_is_capture(Move move)
{
    return (move_flag(move) & FLAG_CAPTURE) != 0;
}

static inline bool move_is_ep(Move move)
{
    return move_flag(move) == FLAG_EP;
}

static inline bool move_is_castle(Move move)
{
    return move_flag(move) == FLAG_CASTLE_KINGSIDE || move_flag(move) == FLAG_CASTLE_QUEENSIDE;
}

static inline bool move_is_double(Move move)
{
    return move_flag(move) == FLAG_PAWN_DOUBLE_PUSH;
}

static inline bool move_is_promo(Move move)
{
    return move_flag(move) >= FLAG_PROMOTION_KNIGHT;
}

static inline bool move_is_quiet(Move move)
{
    return !move_is_capture(move) && !move_is_promo(move);
}

static inline Piece move_promo_pt(Move move)
{
    return (Piece)(KNIGHT + (move_flag(move) & 3));
}

/**
 * @brief Format a move as UCI text ("0000" for none, promotion suffix n/b/r/q).
 * @param move the move to format.
 * @param buf destination, needs >= 8 bytes.
 * @return @p buf (NUL-terminated).
 */
static inline char *move_to_uci(Move move, char *buf)
{
    if (move_is_none(move))
    {
        buf[0] = '0';
        buf[1] = '0';
        buf[2] = '0';
        buf[3] = '0';
        buf[4] = '\0';
        return buf;
    }
    char from_name[3], to_name[3];
    sq_name(move_from(move), from_name);
    sq_name(move_to(move), to_name);
    int length    = 0;
    buf[length++] = from_name[0];
    buf[length++] = from_name[1];
    buf[length++] = to_name[0];
    buf[length++] = to_name[1];
    if (move_is_promo(move))
    {
        buf[length++] = "  nbrq"[move_promo_pt(move)]; // KNIGHT(2)..QUEEN(5) -> n,b,r,q
    }
    buf[length] = '\0';
    return buf;
}

/** @brief Search value scale (centipawns, side-to-move relative). */
enum
{
    MAX_PLY           = 128,                 ///< maximum search depth / ply count
    VALUE_INF         = 32001,               ///< sentinel above any real score
    VALUE_MATE        = 32000,               ///< a mate (distance encoded as VALUE_MATE - ply)
    VALUE_NONE        = 32002,               ///< "no value" sentinel
    VALUE_MATE_IN_MAX = VALUE_MATE - MAX_PLY ///< scores at/above this are forced mates
};

static inline bool is_mate_score(int value)
{
    return value >= VALUE_MATE_IN_MAX || value <= -VALUE_MATE_IN_MAX;
}

/// @name Bitboard file/rank masks and helpers (branch-free via compiler builtins).
/// @{
#define FILE_A ((Bitboard)0x0101010101010101ULL)
#define FILE_B (FILE_A << 1)
#define FILE_G (FILE_A << 6)
#define FILE_H (FILE_A << 7)
#define RANK_1 ((Bitboard)0xffULL)
#define RANK_2 (RANK_1 << 8)
#define RANK_4 (RANK_1 << 24)
#define RANK_5 (RANK_1 << 32)
#define RANK_7 (RANK_1 << 48)
#define RANK_8 (RANK_1 << 56)

static inline Bitboard sq_bb(int square)
{
    return 1ULL << square;
}

static inline int popcount(Bitboard bitboard)
{
    return __builtin_popcountll(bitboard);
}

static inline int lsb(Bitboard bitboard)
{
    return __builtin_ctzll(bitboard);
}

static inline int pop_lsb(Bitboard *bitboard)
{
    int square = lsb(*bitboard);
    *bitboard &= *bitboard - 1;
    return square;
}

static inline bool has_more_than_one(Bitboard bitboard)
{
    return (bitboard & (bitboard - 1)) != 0;
}

/// @}

/**
 * @brief Compass directions as square-index deltas, for the directional shift helpers.
 *
 * One inline shift function per direction (wrap-safe via file masks), keyed by this Direction enum.
 */
typedef enum
{
    NORTH     = 8,
    SOUTH     = -8,
    EAST      = 1,
    WEST      = -1,
    NORTHEAST = 9,
    NORTHWEST = 7,
    SOUTHEAST = -7,
    SOUTHWEST = -9
} Dir;

static inline Bitboard shift_north(Bitboard bitboard)
{
    return bitboard << 8;
}

static inline Bitboard shift_south(Bitboard bitboard)
{
    return bitboard >> 8;
}

static inline Bitboard shift_east(Bitboard bitboard)
{
    return (bitboard & ~FILE_H) << 1;
}

static inline Bitboard shift_west(Bitboard bitboard)
{
    return (bitboard & ~FILE_A) >> 1;
}

static inline Bitboard shift_northeast(Bitboard bitboard)
{
    return (bitboard & ~FILE_H) << 9;
}

static inline Bitboard shift_northwest(Bitboard bitboard)
{
    return (bitboard & ~FILE_A) << 7;
}

static inline Bitboard shift_southeast(Bitboard bitboard)
{
    return (bitboard & ~FILE_H) >> 7;
}

static inline Bitboard shift_southwest(Bitboard bitboard)
{
    return (bitboard & ~FILE_A) >> 9;
}
