// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief The lockless transposition table for Lazy SMP, its payload, and mate-score conversion.
 */
#pragma once
#include "types.h"
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/** @brief The kind of bound a stored score represents. */
typedef enum
{
    BOUND_NONE  = 0, ///< no/empty entry
    BOUND_UPPER = 1, ///< fail-low: score is an upper bound
    BOUND_LOWER = 2, ///< fail-high: score is a lower bound
    BOUND_EXACT = 3  ///< exact score (PV node)
} Bound;

#define TT_MAX_MB 65536 ///< max hash size (MB); must match the advertised `Hash` spin max in uci.c

/**
 * @brief The 64-bit transposition-table payload as a bit-field struct.
 *
 * Field access compiles to a shift/mask, but only for the fields a caller actually reads (no eager unpack),
 * and stores compose the whole word in one go by type-punning through memcpy. All fields use 64-bit base
 * types so the struct is a single 8-byte allocation unit (LSB-first on this ABI); signed bit-fields
 * sign-extend on read.
 */
typedef struct TTData
{
    uint64_t move : 16;  ///< packed Move (0 = none)
    int64_t  score : 16; ///< score_to_tt-adjusted search score
    int64_t  eval : 16;  ///< raw static eval (VALUE_NONE if in check)
    int64_t  depth : 8;  ///< SIGNED [-128,127]: qsearch-style entries (depth <= 0) must not wrap to "deep"
    uint64_t bound : 2;  ///< Bound
    uint64_t gen : 6;    ///< generation the entry was written in
} TTData;

_Static_assert(sizeof(TTData) == 8, "TTData must pack into one 64-bit word");

/** @brief Reinterpret a TTData bit-field as its raw 64-bit word (type-punning via memcpy). */
static inline uint64_t tt_data_to_u64(TTData data)
{
    uint64_t word;
    memcpy(&word, &data, sizeof word);
    return word;
}

/** @brief Reinterpret a raw 64-bit word back into a TTData bit-field. */
static inline TTData u64_to_tt_data(uint64_t word)
{
    TTData data;
    memcpy(&data, &word, sizeof data);
    return data;
}

/**
 * @brief One transposition-table slot, stored as {key ^ data, data} for a lockless torn-read guard.
 *
 * A torn read (data and key from different writes) fails the `key ^ data == probe key` check and is treated
 * as a miss, so threads can probe/store concurrently without locks (occasional benign misses on races). The
 * slot words are _Atomic so the relaxed u64 loads/stores are well-defined.
 */
typedef struct TTSlot
{
    _Atomic uint64_t key;  ///< real key ^ data
    _Atomic uint64_t data; ///< tt_data_to_u64(TTData)
} TTSlot;

/** @brief A transposition table: a power-of-two array of slots with a generation counter. */
typedef struct TranspositionTable
{
    TTSlot  *table;      ///< slot array (calloc'd)
    size_t   slot_count; ///< number of slots (a power of two)
    uint64_t mask;       ///< slot_count - 1, for indexing by key
    uint8_t  generation; ///< current search generation, for aging
} TranspositionTable;

/** @brief (Re)allocate @p tt to @p megabytes (clamped to [1, TT_MAX_MB]); clears it. */
void tt_resize(TranspositionTable *tt, size_t megabytes);
/** @brief Zero every slot of @p tt and reset the generation. */
void tt_clear(TranspositionTable *tt);

/** @brief Advance @p tt's generation so older entries become replaceable. Call once at the start of a search. */
static inline void tt_new_search(TranspositionTable *tt)
{
    tt->generation++;
}

/** @brief Probe @p tt for @p key. @return true on a key hit with a real entry, copying the payload into @p out. */
bool tt_probe(const TranspositionTable *tt, uint64_t key, TTData *out);
/** @brief Store a result under @p key (depth-preferred replacement with generation aging). */
void tt_store(TranspositionTable *tt, uint64_t key, int score, int eval, int depth, Bound bound, Move move, int ply);
/** @brief Approximate fill of @p tt (per mille) over a 1000-slot sample, of the current generation. */
int tt_hashfull(const TranspositionTable *tt);

/**
 * @brief Prefetch @p key's slot into cache to hide the probe's memory latency.
 *
 * Called right after a child is made, so the slot is in flight while the caller finishes this node's work
 * (extensions, history push, reduction) before the recursive search probes it.
 */
static inline void tt_prefetch(const TranspositionTable *tt, uint64_t key)
{
    __builtin_prefetch(&tt->table[key & tt->mask]);
}

/**
 * @brief Adjust a mate score to distance-from-this-node before storing it.
 *
 * Mate scores are stored as distance-from-this-node; convert on the way in/out so a mate found deep in the
 * tree is scored correctly wherever the entry is reused.
 */
static inline int score_to_tt(int score, int ply)
{
    if (score >= VALUE_MATE_IN_MAX)
    {
        return score + ply;
    }
    if (score <= -VALUE_MATE_IN_MAX)
    {
        return score - ply;
    }
    return score;
}

/** @brief Adjust a mate score read from the TT back to distance-from-root. Inverse of score_to_tt(). */
static inline int score_from_tt(int score, int ply)
{
    if (score == VALUE_NONE)
    {
        return score;
    }
    if (score >= VALUE_MATE_IN_MAX)
    {
        return score - ply;
    }
    if (score <= -VALUE_MATE_IN_MAX)
    {
        return score + ply;
    }
    return score;
}
