// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief The PeSTO tapered hand-crafted evaluation and the shared lockless eval cache.
 *
 * PeSTO tapered evaluation (Rofchade's piece-square tables): material + PST interpolated between a
 * middlegame and endgame score by a phase count. Strong, compact, and self-contained; it is the fallback
 * used when no net is loaded — NNUE replaces the whole function behind the evaluate() seam. A few cheap,
 * uncontroversial terms (bishop pair, mobility, tempo) sit on top.
 */
#include "eval.h"
#include "bitboard.h"
#include "nnue.h"
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

static const int mg_value[6]  = {82, 337, 365, 477, 1025, 0};
static const int eg_value[6]  = {94, 281, 297, 512, 936, 0};
static const int phase_inc[6] = {0, 1, 1, 2, 4, 0};

// Tables are printed with index 0 = a8. White reads pst[sq ^ 56]; black reads pst[sq].
static const int mg_pawn[64]   = {0,   0,  0,   0,   0,   0,  0,  0,   98,  134, 61, 95,  68, 126, 34, -11,
                                  -6,  7,  26,  31,  65,  56, 25, -20, -14, 13,  6,  21,  23, 12,  17, -23,
                                  -27, -2, -5,  12,  17,  6,  10, -25, -26, -4,  -4, -10, 3,  3,   33, -12,
                                  -35, -1, -20, -23, -15, 24, 38, -22, 0,   0,   0,  0,   0,  0,   0,  0};
static const int eg_pawn[64]   = {0,  0,   0,  0,  0,  0,  0,  0,  178, 173, 158, 134, 147, 132, 165, 187,
                                  94, 100, 85, 67, 56, 53, 82, 84, 32,  24,  13,  5,   -2,  4,   17,  17,
                                  13, 9,   -3, -7, -7, -8, 3,  -1, 4,   7,   -6,  1,   0,   -5,  -1,  -8,
                                  13, 8,   8,  10, 13, 0,  2,  -7, 0,   0,   0,   0,   0,   0,   0,   0};
static const int mg_knight[64] = {-167, -89, -34, -49, 61, -97, -15, -107, -73,  -41, 72,  36,  23,  62,  7,   -17,
                                  -47,  60,  37,  65,  84, 129, 73,  44,   -9,   17,  19,  53,  37,  69,  18,  22,
                                  -13,  4,   16,  13,  28, 19,  21,  -8,   -23,  -9,  12,  10,  19,  17,  25,  -16,
                                  -29,  -53, -12, -3,  -1, 18,  -14, -19,  -105, -21, -58, -33, -17, -28, -19, -23};
static const int eg_knight[64] = {-58, -38, -13, -28, -31, -27, -63, -99, -25, -8,  -25, -2,  -9,  -25, -24, -52,
                                  -24, -20, 10,  9,   -1,  -9,  -19, -41, -17, 3,   22,  22,  22,  11,  8,   -18,
                                  -18, -6,  16,  25,  16,  17,  4,   -18, -23, -3,  -1,  15,  10,  -3,  -20, -22,
                                  -42, -20, -10, -5,  -2,  -20, -23, -44, -29, -51, -23, -15, -22, -18, -50, -64};
static const int mg_bishop[64] = {-29, 4,  -82, -37, -25, -42, 7,  -8, -26, 16, -18, -13, 30,  59,  18,  -47,
                                  -16, 37, 43,  40,  35,  50,  37, -2, -4,  5,  19,  50,  37,  37,  7,   -2,
                                  -6,  13, 13,  26,  34,  12,  10, 4,  0,   15, 15,  15,  14,  27,  18,  10,
                                  4,   15, 16,  0,   7,   21,  33, 1,  -33, -3, -14, -21, -13, -12, -39, -21};
static const int eg_bishop[64] = {-14, -21, -11, -8, -7, -9, -17, -24, -8,  -4, 7,   -12, -3, -13, -4, -14,
                                  2,   -8,  0,   -1, -2, 6,  0,   4,   -3,  9,  12,  9,   14, 10,  3,  2,
                                  -6,  3,   13,  19, 7,  10, -3,  -9,  -12, -3, 8,   10,  13, 3,   -7, -15,
                                  -14, -18, -7,  -1, 4,  -9, -15, -27, -23, -9, -23, -5,  -9, -16, -5, -17};
static const int mg_rook[64]   = {32,  42,  32,  51, 63, 9,  31, 43,  27,  32,  58,  62,  80, 67, 26,  44,
                                  -5,  19,  26,  36, 17, 45, 61, 16,  -24, -11, 7,   26,  24, 35, -8,  -20,
                                  -36, -26, -12, -1, 9,  -7, 6,  -23, -45, -25, -16, -17, 3,  0,  -5,  -33,
                                  -44, -16, -20, -9, -1, 11, -6, -71, -19, -13, 1,   17,  16, 7,  -37, -26};
static const int eg_rook[64] = {13, 10,  18, 15,  12, 12, 8, 5, 11, 13, 13,  11, -3, 3, 8,  3,  7,  7,   7,  5,  4,  -3,
                                -5, -3,  4,  3,   13, 1,  2, 1, -1, 2,  3,   5,  8,  4, -5, -6, -8, -11, -4, 0,  -5, -1,
                                -7, -12, -8, -16, -6, -6, 0, 2, -9, -9, -11, -3, -9, 2, 3,  -1, -5, -13, 4,  -20};
static const int mg_queen[64] = {-28, 0,   29, 12,  59, 44, 43, 45, -24, -39, -5,  1,   -16, 57,  28,  54,
                                 -13, -17, 7,  8,   29, 56, 47, 57, -27, -27, -16, -16, -1,  17,  -2,  1,
                                 -9,  -26, -9, -10, -2, -4, 3,  -3, -14, 2,   -11, -2,  -5,  2,   14,  5,
                                 -35, -8,  11, 2,   8,  15, -3, 1,  -1,  -18, -9,  10,  -15, -25, -31, -50};
static const int eg_queen[64] = {-9,  22,  22,  27,  27,  19,  10,  20,  -17, 20,  32,  41,  58, 25,  30,  0,
                                 -20, 6,   9,   49,  47,  35,  19,  9,   3,   22,  24,  45,  57, 40,  57,  36,
                                 -18, 28,  19,  47,  31,  34,  39,  23,  -16, -27, 15,  6,   9,  17,  10,  5,
                                 -22, -23, -30, -16, -16, -23, -36, -32, -33, -28, -22, -43, -5, -32, -20, -41};
static const int mg_king[64]  = {-65, 23, 16,  -15, -56, -34, 2,   13,  29,  -1,  -20, -7,  -8,  -4,  -38, -29,
                                 -9,  24, 2,   -16, -20, 6,   22,  -22, -17, -20, -12, -27, -30, -25, -14, -36,
                                 -49, -1, -27, -39, -46, -44, -33, -51, -14, -14, -22, -46, -44, -30, -15, -27,
                                 1,   7,  -8,  -64, -43, -16, 9,   8,   -15, 36,  12,  -54, 8,   -28, 24,  14};
static const int eg_king[64]  = {-74, -35, -18, -18, -11, 15, 4,  -17, -12, 17,  14,  17,  17,  38,  23,  11,
                                 10,  17,  23,  15,  20,  45, 44, 13,  -8,  22,  24,  27,  26,  33,  26,  3,
                                 -18, -4,  21,  24,  27,  23, 9,  -11, -19, -3,  11,  21,  23,  16,  7,   -9,
                                 -27, -11, 4,   13,  14,  4,  -5, -17, -53, -34, -21, -11, -28, -14, -24, -43};

static const int *mg_pst[6] = {mg_pawn, mg_knight, mg_bishop, mg_rook, mg_queen, mg_king};
static const int *eg_pst[6] = {eg_pawn, eg_knight, eg_bishop, eg_rook, eg_queen, eg_king};

static int mg_table[12][64];
static int eg_table[12][64];

// Shared lockless eval cache: 2^20 single-u64 entries = 8 MB. Each entry packs the key's high 48 bits with
// the 16-bit eval; the slot index uses the key's low 20 bits, so a verified hit implies ALL 64 key bits
// match (bits 0-19 via the index, 16-63 via the compare) — false hits are impossible, a collision only
// evicts. Single-word relaxed atomics cannot tear, so it is Lazy-SMP-safe like the TT. The big win is
// qsearch, which evaluates at every stand-pat with no other caching.
#define EVAL_CACHE_ENTRIES (1ull << 20)
static _Atomic uint64_t *eval_cache;

static uint64_t eval_cache_pack(const uint64_t key, const int value)
{
    return (key & ~0xFFFFull) | (uint16_t)(int16_t)value;
}

void init_eval(void)
{
    for (int piece = 0; piece < 6; piece++)
    {
        for (int square = 0; square < 64; square++)
        {
            mg_table[piece][square]     = mg_value[piece] + mg_pst[piece][square ^ 56]; // white
            eg_table[piece][square]     = eg_value[piece] + eg_pst[piece][square ^ 56];
            mg_table[piece + 6][square] = mg_value[piece] + mg_pst[piece][square]; // black
            eg_table[piece + 6][square] = eg_value[piece] + eg_pst[piece][square];
        }
    }
    // The eval cache is a zero-initialised table allocated on first init.
    if (eval_cache == NULL)
    {
        eval_cache = calloc(EVAL_CACHE_ENTRIES, sizeof(_Atomic uint64_t));
    }
}

void eval_cache_clear(void)
{
    if (eval_cache)
    {
        memset((void *)eval_cache, 0, EVAL_CACHE_ENTRIES * sizeof(_Atomic uint64_t));
    }
}

/** @brief Accumulate a mobility term: attacked squares not occupied by own pieces, small weights. */
static void mobility(int *middlegame, int *endgame, Bitboard piece_bitboard, Bitboard (*attack_fn)(int, Bitboard),
                     const Bitboard occupancy, const Bitboard own_pieces, const int middlegame_weight,
                     const int endgame_weight)
{
    while (piece_bitboard)
    {
        const int attacker_square = pop_lsb(&piece_bitboard);
        const int mobility_count  = popcount(attack_fn(attacker_square, occupancy) & ~own_pieces);
        *middlegame += middlegame_weight * (mobility_count - 4);
        *endgame += endgame_weight * (mobility_count - 4);
    }
}

int evaluate(const Position *pos)
{
    // eval_cache is NULL only if its allocation failed at init — degrade to an uncached eval rather than
    // dereferencing NULL. The branch is perfectly predicted (cache is non-NULL in every normal run).
    _Atomic uint64_t *const slot = eval_cache ? &eval_cache[pos->key & (EVAL_CACHE_ENTRIES - 1)] : NULL;
    if (slot)
    {
        const uint64_t entry = atomic_load_explicit(slot, memory_order_relaxed);
        if (entry != 0 && ((entry ^ pos->key) & ~0xFFFFull) == 0)
        {
            return (int16_t)(uint16_t)entry; // hit: the low 16 bits hold the cached eval
        }
    }

    int value;
    // NNUE replaces the whole hand-crafted evaluation when a net is loaded (UCI EvalFile). Read the
    // incrementally-maintained accumulator (kept in sync by make_move/set_fen) — a cheap forward pass.
    if (nnue_is_loaded())
    {
        value = nnue_evaluate(&pos->accumulator, pos->color_to_move);
        if (slot)
        {
            atomic_store_explicit(slot, eval_cache_pack(pos->key, value), memory_order_relaxed);
        }
        return value;
    }

    int middlegame[2] = {0, 0}, endgame[2] = {0, 0}, phase = 0;

    const Bitboard occupancy  = position_occupied(pos);
    Bitboard       board_bits = occupancy;
    while (board_bits)
    {
        const int   square = pop_lsb(&board_bits);
        const Piece piece  = (Piece)pos->board[square];
        const Color color  = position_color_on(pos, square);
        // The PST tables are laid out as 12 rows: white types 0..5, then black types 6..11 (0-based types).
        const int code = color * 6 + (int)piece - 1;
        middlegame[color] += mg_table[code][square];
        endgame[color] += eg_table[code][square];
        phase += phase_inc[(int)piece - 1];
    }

    // Cheap extra terms.
    for (int color = WHITE; color <= BLACK; color++)
    {
        // Bishop pair.
        if (popcount(position_pieces(pos, (Color)color, BISHOP)) >= 2)
        {
            middlegame[color] += 22;
            endgame[color] += 40;
        }
        // Mobility (attacked squares not occupied by own pieces), small weights.
        const Bitboard own_pieces = pos->colors[color];
        mobility(&middlegame[color], &endgame[color], position_pieces(pos, (Color)color, BISHOP), bishop_attacks,
                 occupancy, own_pieces, 4, 4);
        mobility(&middlegame[color], &endgame[color], position_pieces(pos, (Color)color, ROOK), rook_attacks, occupancy,
                 own_pieces, 2, 4);
        mobility(&middlegame[color], &endgame[color], position_pieces(pos, (Color)color, QUEEN), queen_attacks,
                 occupancy, own_pieces, 1, 2);
    }

    const int middlegame_score = middlegame[WHITE] - middlegame[BLACK];
    const int endgame_score    = endgame[WHITE] - endgame[BLACK];
    if (phase > 24)
    {
        phase = 24;
    }
    const int score              = (middlegame_score * phase + endgame_score * (24 - phase)) / 24; // White-relative
    const int side_to_move_score = (pos->color_to_move == WHITE ? score : -score);
    value                        = side_to_move_score + 10; // tempo: a small bonus for the side to move
    if (slot)
    {
        atomic_store_explicit(slot, eval_cache_pack(pos->key, value), memory_order_relaxed);
    }
    return value;
}
