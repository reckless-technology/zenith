// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief Precomputed attack tables and magic-bitboard sliding attacks.
 *
 * The leaper/geometry tables (pawn/knight/king attacks, BetweenBB/LineBB) and the 128 magic MULTIPLIERS are
 * generated compile-time constants (tools/generate_tables.py). The magic sliding-attack tables themselves are
 * the engine's ONE remaining piece of mutable file-scope state: init_bitboards() fills them deterministically
 * from the constant multipliers exactly once at startup — no search, no PRNG — before any thread exists, and
 * they are read-only ever after. They stay file-scope rather than
 * living in the Engine because the magic lookups are the hottest loads in the engine, and threading a
 * context pointer through them would tax every sliding-attack call for zero practical benefit; at ~850KB
 * they are also impractical to bake into source (unlike these geometry tables).
 */
#pragma once
#include "types.h"

// Generated compile-time constants (tools/generate_tables.py -> src/bitboard_tables.inc).
extern const Bitboard PawnAttacks[NUM_COLORS][64]; ///< pawn attacks by [color][square]
extern const Bitboard KnightAttacks[64];           ///< knight attacks by square
extern const Bitboard KingAttacks[64];             ///< king attacks by square
extern const Bitboard BetweenBB[64][64];           ///< squares strictly between two aligned squares (exclusive), else 0
extern const Bitboard LineBB[64][64];              ///< the whole rank/file/diagonal through two aligned squares, else 0

/** @brief Bishop sliding attacks from @p square given @p occupancy (magic-bitboard lookup). */
Bitboard bishop_attacks(int square, Bitboard occupancy);
/** @brief Rook sliding attacks from @p square given @p occupancy (magic-bitboard lookup). */
Bitboard rook_attacks(int square, Bitboard occupancy);

/** @brief Queen sliding attacks (bishop | rook) from @p square given @p occupancy. */
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

/** @brief Fill the magic sliding-attack tables from the generated multipliers. Call once at startup. */
void init_bitboards(void);
