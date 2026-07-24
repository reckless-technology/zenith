// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
#pragma once
#include "types.h"

extern Bitboard PawnAttacks[COLOR_NB][64];
extern Bitboard KnightAttacks[64];
extern Bitboard KingAttacks[64];
extern Bitboard BetweenBB[64][64]; // squares strictly between two aligned squares (exclusive), else 0
extern Bitboard LineBB[64][64];    // the whole rank/file/diagonal through two aligned squares, else 0

// Sliding attacks via magic bitboards (tables filled by init_bitboards()).
Bitboard bishop_attacks(int square, Bitboard occupancy);
Bitboard rook_attacks(int square, Bitboard occupancy);

static inline Bitboard queen_attacks(int square, Bitboard occupancy)
{
    return bishop_attacks(square, occupancy) | rook_attacks(square, occupancy);
}

static inline Bitboard knight_attacks(int square)
{
    return KnightAttacks[square];
}

static inline Bitboard king_attacks(int square)
{
    return KingAttacks[square];
}

static inline Bitboard pawn_attacks(Color color, int square)
{
    return PawnAttacks[color][square];
}

static inline Bitboard rank_bb(int square)
{
    return RANK_1 << (8 * rank_of(square));
}

static inline Bitboard file_bb(int square)
{
    return FILE_A << file_of(square);
}

static inline Bitboard between_bb(int from, int to)
{
    return BetweenBB[from][to];
}

static inline Bitboard line_bb(int from, int to)
{
    return LineBB[from][to];
}

void init_bitboards(void);
