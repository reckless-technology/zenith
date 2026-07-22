#pragma once
#include "types.h"

extern Bitboard PawnAttacks[COLOR_NB][64];
extern Bitboard KnightAttacks[64];
extern Bitboard KingAttacks[64];
extern Bitboard BetweenBB[64][64]; // squares strictly between two aligned squares (exclusive), else 0
extern Bitboard LineBB[64][64];    // the whole rank/file/diagonal through two aligned squares, else 0

// Sliding attacks via magic bitboards (tables filled by init_bitboards()).
Bitboard bishop_attacks(int sq, Bitboard occ);
Bitboard rook_attacks(int sq, Bitboard occ);

inline Bitboard queen_attacks(int sq, Bitboard occ)
{
    return bishop_attacks(sq, occ) | rook_attacks(sq, occ);
}

inline Bitboard knight_attacks(int sq)
{
    return KnightAttacks[sq];
}

inline Bitboard king_attacks(int sq)
{
    return KingAttacks[sq];
}

inline Bitboard pawn_attacks(Color c, int sq)
{
    return PawnAttacks[c][sq];
}

inline Bitboard rank_bb(int sq)
{
    return RANK_1 << (8 * rank_of(sq));
}

inline Bitboard file_bb(int sq)
{
    return FILE_A << file_of(sq);
}

inline Bitboard between_bb(int a, int b)
{
    return BetweenBB[a][b];
}

inline Bitboard line_bb(int a, int b)
{
    return LineBB[a][b];
}

void init_bitboards();
