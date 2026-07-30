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
#ifdef ZENITH_USE_PEXT
#include <immintrin.h> // _pext_u64 (BMI2)
#endif

// Generated compile-time constants (tools/generate_tables.py -> src/generated/bitboard_tables.inc).
extern const Bitboard PawnAttacks[NUM_COLORS][64]; ///< pawn attacks by [color][square]
extern const Bitboard KnightAttacks[64];           ///< knight attacks by square
extern const Bitboard KingAttacks[64];             ///< king attacks by square
extern const Bitboard BetweenBB[64][64];           ///< squares strictly between two aligned squares (exclusive), else 0
extern const Bitboard LineBB[64][64];              ///< the whole rank/file/diagonal through two aligned squares, else 0

/** @brief One square's magic-lookup entry: masked-occupancy hashing into its slice of the attack table. */
typedef struct Magic
{
    Bitboard  mask;    ///< relevant occupancy bits for this square
    Bitboard  magic;   ///< the multiplier (offline-generated constant; unused in the PEXT build)
    Bitboard *attacks; ///< this square's slice of the shared attack table
    unsigned  shift;   ///< 64 - popcount(mask): the multiply-shift index width
} Magic;

/// The per-square magic entries, filled once by init_bitboards() (the ~850KB attack tables stay
/// file-scope in bitboard.c); exposed only so the lookups below can inline into the movegen hot path.
extern Magic RookMagics[64];
extern Magic BishopMagics[64];

/** @brief Dense index of @p occupancy within @p entry's attack-table slice. */
static inline unsigned magic_index(const Magic *entry, const Bitboard occupancy)
{
#ifdef ZENITH_USE_PEXT
    // BMI2 parallel-bit-extract: pack the masked occupancy bits into a dense index directly, no multiply.
    return (unsigned)_pext_u64(occupancy, entry->mask);
#else
    return (unsigned)(((occupancy & entry->mask) * entry->magic) >> entry->shift);
#endif
}

/** @brief Bishop sliding attacks from @p square given @p occupancy (magic-bitboard lookup).
 *  Inline: this is the hottest load in move generation — no LTO required to keep it call-free. */
static inline Bitboard bishop_attacks(const int square, const Bitboard occupancy)
{
    const Magic *const entry = &BishopMagics[square];
    return entry->attacks[magic_index(entry, occupancy)];
}

/** @brief Rook sliding attacks from @p square given @p occupancy (magic-bitboard lookup). Inline, as above. */
static inline Bitboard rook_attacks(const int square, const Bitboard occupancy)
{
    const Magic *const entry = &RookMagics[square];
    return entry->attacks[magic_index(entry, occupancy)];
}

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

/** @brief Fill the magic sliding-attack tables from the generated multipliers. Idempotent; called
 *  lazily by engine_new(), so no caller needs an explicit startup step. */
void init_bitboards(void);
