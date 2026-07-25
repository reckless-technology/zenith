// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief Attack-table construction: leaper tables and startup-generated magic bitboards.
 */
#include "bitboard.h"
#include <stdlib.h>

Bitboard PawnAttacks[COLOR_NB][64];
Bitboard KnightAttacks[64];
Bitboard KingAttacks[64];
Bitboard BetweenBB[64][64];
Bitboard LineBB[64][64];

// Deterministic sparse PRNG for magic search (xorshift64*), seeded once.
typedef struct PRNG
{
    uint64_t state;
} PRNG;

static uint64_t prng_next(PRNG *prng)
{
    prng->state ^= prng->state >> 12;
    prng->state ^= prng->state << 25;
    prng->state ^= prng->state >> 27;
    return prng->state * 0x2545F4914F6CDD1DULL;
}

static uint64_t prng_sparse(PRNG *prng)
{
    return prng_next(prng) & prng_next(prng) & prng_next(prng); // few set bits -> good magic candidates
}

/** @brief Ray-walk sliding attacks from @p square (used to build masks and to fill the magic tables). */
static Bitboard sliding_attack(int square, Bitboard occupancy, const int deltas[4])
{
    Bitboard attacks = 0;
    for (int direction = 0; direction < 4; direction++)
    {
        int delta   = deltas[direction];
        int current = square;
        while (true)
        {
            int previous_file = file_of(current);
            int next_square   = current + delta;
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
static const int BishopDirs[4] = {NE, NW, SE, SW};

typedef struct Magic
{
    Bitboard  mask;
    Bitboard  magic;
    Bitboard *attacks;
    unsigned  shift;
} Magic;

static unsigned magic_index(const Magic *entry, Bitboard occupancy)
{
    return (unsigned)(((occupancy & entry->mask) * entry->magic) >> entry->shift);
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
static void init_magics(bool is_rook, Bitboard *table, Magic magics[64], const int deltas[4])
{
    Bitboard  occupancy[4096], reference[4096];
    int       epoch[4096]   = {0};
    int       epoch_counter = 0;
    Bitboard *attack_base   = table;

    for (int square = 0; square < 64; square++)
    {
        // Relevant-occupancy mask = empty-board rays minus the edges not on the piece's own rank/file.
        Bitboard edges         = ((RANK_1 | RANK_8) & ~rank_bb(square)) | ((FILE_A | FILE_H) & ~file_bb(square));
        Bitboard mask          = sliding_attack(square, 0, deltas) & ~edges;
        unsigned relevant_bits = popcount(mask);
        magics[square].mask    = mask;
        magics[square].shift   = 64 - relevant_bits;
        magics[square].attacks = attack_base;

        // Enumerate every subset of mask (Carry-Rippler), recording its true attack set.
        Bitboard subset       = 0;
        int      subset_count = 0;
        do
        {
            occupancy[subset_count] = subset;
            reference[subset_count] = sliding_attack(square, subset, deltas);
            subset_count++;
            subset = (subset - mask) & mask;
        } while (subset);

        PRNG prng = {0x9E3779B97F4A7C15ULL ^ ((uint64_t)square * 0xBF58476D1CE4E5B9ULL) ^ (is_rook ? 1 : 2)};
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
                unsigned table_index = magic_index(&magics[square], occupancy[subset_index]);
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
        attack_base += subset_count;
    }
}

Bitboard bishop_attacks(int square, Bitboard occupancy)
{
    const Magic *entry = &BishopMagics[square];
    return entry->attacks[magic_index(entry, occupancy)];
}

Bitboard rook_attacks(int square, Bitboard occupancy)
{
    const Magic *entry = &RookMagics[square];
    return entry->attacks[magic_index(entry, occupancy)];
}

void init_bitboards(void)
{
    for (int square = 0; square < 64; square++)
    {
        Bitboard square_bit        = sq_bb(square);
        PawnAttacks[WHITE][square] = shift_ne(square_bit) | shift_nw(square_bit);
        PawnAttacks[BLACK][square] = shift_se(square_bit) | shift_sw(square_bit);

        // Knight: all (±1,±2)/(±2,±1) offsets, rejecting wraps by file/rank distance.
        Bitboard  knight_attack = 0, king_attack = 0;
        int       file = file_of(square), rank = rank_of(square);
        const int knight_file[8] = {1, 2, 2, 1, -1, -2, -2, -1};
        const int knight_rank[8] = {2, 1, -1, -2, -2, -1, 1, 2};
        const int king_file[8]   = {0, 1, 1, 1, 0, -1, -1, -1};
        const int king_rank[8]   = {1, 1, 0, -1, -1, -1, 0, 1};
        for (int offset = 0; offset < 8; offset++)
        {
            int target_file = file + knight_file[offset], target_rank = rank + knight_rank[offset];
            if (target_file >= 0 && target_file < 8 && target_rank >= 0 && target_rank < 8)
            {
                knight_attack |= sq_bb(make_square(target_file, target_rank));
            }
            target_file = file + king_file[offset], target_rank = rank + king_rank[offset];
            if (target_file >= 0 && target_file < 8 && target_rank >= 0 && target_rank < 8)
            {
                king_attack |= sq_bb(make_square(target_file, target_rank));
            }
        }
        KnightAttacks[square] = knight_attack;
        KingAttacks[square]   = king_attack;
    }

    init_magics(false, BishopTable, BishopMagics, BishopDirs);
    init_magics(true, RookTable, RookMagics, RookDirs);

    // BetweenBB: squares strictly between a and b when they share a rank/file/diagonal.
    // LineBB: the whole rank/file/diagonal through a and b (endpoints included), 0 if not aligned.
    Bitboard (*const attackers[2])(int, Bitboard) = {rook_attacks, bishop_attacks};
    for (int from = 0; from < 64; from++)
    {
        for (int to = 0; to < 64; to++)
        {
            BetweenBB[from][to] = 0;
            LineBB[from][to]    = 0;
            if (from == to)
            {
                continue;
            }
            for (int slider = 0; slider < 2; slider++)
            {
                Bitboard (*const attacker)(int, Bitboard) = attackers[slider];
                if (attacker(from, 0) & sq_bb(to))
                {
                    BetweenBB[from][to] = attacker(from, sq_bb(to)) & attacker(to, sq_bb(from));
                    LineBB[from][to]    = (sq_bb(from) | attacker(from, 0)) & (sq_bb(to) | attacker(to, 0));
                }
            }
        }
    }
}
