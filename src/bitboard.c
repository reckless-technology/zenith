// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief Magic-bitboard construction (the leaper/geometry tables are generated compile-time constants).
 */
#include "bitboard.h"
#include <assert.h>
#include <stdlib.h>

#ifdef ZENITH_USE_PEXT
#include <immintrin.h> // _pext_u64 (BMI2)
#endif

// The leaper/geometry tables are generated constants (see tools/generate_tables.py); only the magic
// sliding-attack tables below are runtime-FILLED (their multipliers are generated constants too).
#include "generated/bitboard_tables.inc"
#ifndef ZENITH_USE_PEXT
#include "generated/magic_tables.inc"
#endif

/** @brief Ray-walk sliding attacks from @p square (used to build masks and to fill the magic tables). */
static Bitboard sliding_attack(const int square, const Bitboard occupancy, const int deltas[4])
{
    Bitboard attacks = 0;
    for (int direction = 0; direction < 4; direction++)
    {
        const int delta   = deltas[direction];
        int       current = square;
        while (true)
        {
            const int previous_file = file_of(current);
            const int next_square   = current + delta;
            if (next_square < 0 || next_square >= 64)
            {
                break;
            }
            // reject rank/file wraps: any king-step move changes file by at most 1
            if (abs(file_of(next_square) - previous_file) > 1)
            {
                break;
            }
            attacks |= sq_bb(next_square);
            if (occupancy & sq_bb(next_square))
            {
                break; // blocker occupies this square; stop after including it
            }
            current = next_square;
        }
    }
    return attacks;
}

static const int RookDirs[4]   = {NORTH, SOUTH, EAST, WEST};
static const int BishopDirs[4] = {NORTHEAST, NORTHWEST, SOUTHEAST, SOUTHWEST};

typedef struct Magic
{
    Bitboard  mask;
    Bitboard  magic;
    Bitboard *attacks;
    unsigned  shift;
} Magic;

static unsigned magic_index(const Magic *entry, const Bitboard occupancy)
{
#ifdef ZENITH_USE_PEXT
    // BMI2 parallel-bit-extract: pack the masked occupancy bits into a dense index directly, no multiply.
    return (unsigned)_pext_u64(occupancy, entry->mask);
#else
    return (unsigned)(((occupancy & entry->mask) * entry->magic) >> entry->shift);
#endif
}

static Magic    RookMagics[64];
static Magic    BishopMagics[64];
static Bitboard RookTable[0x19000];  // 102400
static Bitboard BishopTable[0x1480]; // 5248

/**
 * @brief Find a magic multiplier per square and fill the attack table for one slider kind.
 *
 * For each square: build the relevant-occupancy mask, enumerate every occupancy subset (Carry-Rippler),
 * then search sparse random magics until one maps every subset to a collision-free index (an epoch counter
 * distinguishes "not yet written this attempt" from a real collision).
 */
static void init_magics(const Bitboard magic_numbers[64], Bitboard *table, Magic magics[64], const int deltas[4])
{
    Bitboard *attack_base = table;

    for (int square = 0; square < 64; square++)
    {
        // Relevant-occupancy mask = empty-board rays minus the edges not on the piece's own rank/file.
        const Bitboard edges         = ((RANK_1 | RANK_8) & ~rank_bb(square)) | ((FILE_A | FILE_H) & ~file_bb(square));
        const Bitboard mask          = sliding_attack(square, 0, deltas) & ~edges;
        const unsigned relevant_bits = popcount(mask);
        magics[square].mask          = mask;
        magics[square].shift         = 64 - relevant_bits;
        magics[square].attacks       = attack_base;
#ifdef ZENITH_USE_PEXT
        magics[square].magic = 0; // unused with PEXT indexing
        (void)magic_numbers;
#else
        // The multiplier is a generated constant (tools/generate_tables.py replicates the historical
        // search bit-for-bit), so startup just fills the table — no search, no PRNG.
        magics[square].magic = magic_numbers[square];
#endif

        // Enumerate every subset of mask (Carry-Rippler) and fill its slot with the true attack set. The
        // debug build asserts the generated magic is collision-free (a bad table would corrupt the fill).
        Bitboard subset       = 0;
        int      subset_count = 0;
        do
        {
            const Bitboard attacks     = sliding_attack(square, subset, deltas);
            const unsigned table_index = magic_index(&magics[square], subset);
            assert(attack_base[table_index] == 0 || attack_base[table_index] == attacks);
            attack_base[table_index] = attacks;
            subset_count++;
            subset = (subset - mask) & mask;
        } while (subset);

        attack_base += subset_count;
    }
}

Bitboard bishop_attacks(const int square, const Bitboard occupancy)
{
    const Magic *const entry = &BishopMagics[square];
    return entry->attacks[magic_index(entry, occupancy)];
}

Bitboard rook_attacks(const int square, const Bitboard occupancy)
{
    const Magic *const entry = &RookMagics[square];
    return entry->attacks[magic_index(entry, occupancy)];
}

void init_bitboards(void)
{
#ifdef ZENITH_USE_PEXT
    init_magics(NULL, BishopTable, BishopMagics, BishopDirs);
    init_magics(NULL, RookTable, RookMagics, RookDirs);
#else
    init_magics(BishopMagicNumbers, BishopTable, BishopMagics, BishopDirs);
    init_magics(RookMagicNumbers, RookTable, RookMagics, RookDirs);
#endif
}
