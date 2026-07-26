// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief Magic-bitboard construction (the leaper/geometry tables are generated compile-time constants).
 */
#include "bitboard.h"
#include <stdlib.h>

#ifdef ZENITH_USE_PEXT
#include <immintrin.h> // _pext_u64 (BMI2)
#endif

// The leaper/geometry tables are generated constants (see tools/generate_tables.py); only the magic
// sliding-attack tables below are runtime-built.
#include "bitboard_tables.inc"

#ifndef ZENITH_USE_PEXT
// Deterministic sparse PRNG for the magic search (xorshift64*); state is a single uint64_t. Unused with PEXT.
static uint64_t prng_next(uint64_t *state)
{
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return *state * 0x2545F4914F6CDD1DULL;
}

static uint64_t prng_sparse(uint64_t *state)
{
    return prng_next(state) & prng_next(state) & prng_next(state); // few set bits -> good magic candidates
}
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
static void init_magics(const bool is_rook, Bitboard *table, Magic magics[64], const int deltas[4])
{
#ifndef ZENITH_USE_PEXT
    Bitboard occupancy[4096], reference[4096];
    int      epoch[4096]   = {0};
    int      epoch_counter = 0;
#endif
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

        Bitboard subset       = 0;
        int      subset_count = 0;
#ifdef ZENITH_USE_PEXT
        // PEXT: pext(subset, mask) is already a dense index in [0, 2^relevant_bits) — no magic search.
        // Fill the table directly; the same per-square layout as the magic path (attack_base advances by count).
        magics[square].magic = 0; // unused with PEXT indexing
        do
        {
            attack_base[(unsigned)_pext_u64(subset, mask)] = sliding_attack(square, subset, deltas);
            subset_count++;
            subset = (subset - mask) & mask;
        } while (subset);
#else
        // Enumerate every subset of mask (Carry-Rippler), recording its true attack set.
        do
        {
            occupancy[subset_count] = subset;
            reference[subset_count] = sliding_attack(square, subset, deltas);
            subset_count++;
            subset = (subset - mask) & mask;
        } while (subset);

        uint64_t prng = 0x9E3779B97F4A7C15ULL ^ ((uint64_t)square * 0xBF58476D1CE4E5B9ULL) ^ (is_rook ? 1 : 2);
        for (int subset_index = 0; subset_index < subset_count;)
        {
            magics[square].magic = 0;
            // Require the top byte of (mask*magic) to be well-spread, else the magic is poor.
            while (popcount((mask * magics[square].magic) >> 56) < 6)
            {
                magics[square].magic = prng_sparse(&prng);
            }

            epoch_counter++;
            for (subset_index = 0; subset_index < subset_count; subset_index++)
            {
                const unsigned table_index = magic_index(&magics[square], occupancy[subset_index]);
                if (epoch[table_index] < epoch_counter)
                {
                    epoch[table_index]       = epoch_counter;
                    attack_base[table_index] = reference[subset_index];
                }
                else if (attack_base[table_index] != reference[subset_index])
                {
                    break; // index collision with a different attack set -> reject this magic
                }
            }
        }
#endif
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
    init_magics(false, BishopTable, BishopMagics, BishopDirs);
    init_magics(true, RookTable, RookMagics, RookDirs);
}
