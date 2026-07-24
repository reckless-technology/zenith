// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
#pragma once
// Core value types: colors, pieces, squares, and the packed 16-bit move.
#include <stdbool.h>
#include <stdint.h>

typedef uint64_t Bitboard;

typedef enum
{
    WHITE,
    BLACK,
    COLOR_NB = 2
} Color;

static inline Color color_flip(Color c)
{
    return (Color)(c ^ 1);
}

typedef enum
{
    PAWN,
    KNIGHT,
    BISHOP,
    ROOK,
    QUEEN,
    KING,
    PIECE_TYPE_NB = 6,
    NO_PIECE_TYPE = 6
} PieceType;

// Mailbox piece code = color*6 + type; NO_PIECE = 12.
typedef enum
{
    NO_PIECE = 12
} Piece;

static inline Piece make_piece(Color color, PieceType piece_type)
{
    return (Piece)(color * 6 + piece_type);
}

static inline Color color_of(Piece piece)
{
    return (Color)(piece / 6);
}

static inline PieceType type_of(Piece piece)
{
    return (PieceType)(piece % 6);
}

// Squares: A1 = 0 … H8 = 63; rank = sq/8, file = sq%8.
enum
{
    NO_SQ     = 64,
    SQUARE_NB = 64
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

// Writes "-" for off-board squares, else e.g. "e4"; returns buf (NUL-terminated, needs >= 3 bytes).
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

// Castling rights bitmask.
enum
{
    CR_WK  = 1,
    CR_WQ  = 2,
    CR_BK  = 4,
    CR_BQ  = 8,
    CR_ALL = 15
};

// Move flags (CPW encoding): from(0..5) | to(6..11) | flag(12..15).
enum
{
    FLAG_QUIET       = 0,
    FLAG_DOUBLE      = 1,
    FLAG_KCASTLE     = 2,
    FLAG_QCASTLE     = 3,
    FLAG_CAPTURE     = 4,
    FLAG_EP          = 5,
    FLAG_PROMO_N     = 8,
    FLAG_PROMO_B     = 9,
    FLAG_PROMO_R     = 10,
    FLAG_PROMO_Q     = 11,
    FLAG_PROMO_CAP_N = 12,
    FLAG_PROMO_CAP_Q = 15,
};

// A move is a bare uint16 (from | to<<6 | flag<<12); MOVE_NONE (0) is "no move". Being a plain integer
// typedef keeps == / != working directly where the C++ class had operator overloads.
typedef uint16_t Move;

#define MOVE_NONE ((Move)0)

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
    return (move_flag(move) & 4) != 0;
}

static inline bool move_is_ep(Move move)
{
    return move_flag(move) == FLAG_EP;
}

static inline bool move_is_castle(Move move)
{
    return move_flag(move) == FLAG_KCASTLE || move_flag(move) == FLAG_QCASTLE;
}

static inline bool move_is_double(Move move)
{
    return move_flag(move) == FLAG_DOUBLE;
}

static inline bool move_is_promo(Move move)
{
    return move_flag(move) >= FLAG_PROMO_N;
}

static inline bool move_is_quiet(Move move)
{
    return !move_is_capture(move) && !move_is_promo(move);
}

static inline PieceType move_promo_pt(Move move)
{
    return (PieceType)(KNIGHT + (move_flag(move) & 3));
}

// UCI text of a move ("0000" for none, promotion suffix n/b/r/q); returns buf (needs >= 8 bytes).
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
        buf[length++] = "  nbrq"[move_promo_pt(move) + 1]; // KNIGHT..QUEEN -> n,b,r,q
    }
    buf[length] = '\0';
    return buf;
}

// Search value scale.
enum
{
    MAX_PLY           = 128,
    VALUE_INF         = 32001,
    VALUE_MATE        = 32000,
    VALUE_NONE        = 32002,
    VALUE_MATE_IN_MAX = VALUE_MATE - MAX_PLY // scores at/above this are forced mates
};

static inline bool is_mate_score(int value)
{
    return value >= VALUE_MATE_IN_MAX || value <= -VALUE_MATE_IN_MAX;
}

// Bitboard helpers (branch-free via compiler builtins).
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

static inline bool more_than_one(Bitboard bitboard)
{
    return (bitboard & (bitboard - 1)) != 0;
}

// Directional shifts (compass; wrap-safe via file masks). The C++ shift<Dir> template becomes one inline
// function per direction, preserving the semantics exactly.
typedef enum
{
    NORTH = 8,
    SOUTH = -8,
    EAST  = 1,
    WEST  = -1,
    NE    = 9,
    NW    = 7,
    SE    = -7,
    SW    = -9
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

static inline Bitboard shift_ne(Bitboard bitboard)
{
    return (bitboard & ~FILE_H) << 9;
}

static inline Bitboard shift_nw(Bitboard bitboard)
{
    return (bitboard & ~FILE_A) << 7;
}

static inline Bitboard shift_se(Bitboard bitboard)
{
    return (bitboard & ~FILE_H) >> 7;
}

static inline Bitboard shift_sw(Bitboard bitboard)
{
    return (bitboard & ~FILE_A) >> 9;
}
