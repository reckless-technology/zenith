// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief The evaluate() seam (NNUE forward pass) and the shared lockless eval cache.
 *
 * Every position is evaluated by the NNUE — an engine always holds a net (the build-time-embedded one by
 * default, or an EvalFile). The former hand-crafted PeSTO fallback was removed once the embedded net made
 * a netless engine impossible; see git history (tag cpp-final onward) for the tapered HCE.
 */
#include "eval.h"
#include "nnue.h"
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

// The eval cache type lives in eval.h; the big win is qsearch, which evaluates at every stand-pat with no
// other caching.
#define EVAL_CACHE_ENTRIES (1ull << 20)

static uint64_t eval_cache_pack(const uint64_t key, const int value)
{
    return (key & ~0xFFFFull) | (uint16_t)(int16_t)value;
}

void eval_cache_init(EvalCache *cache)
{
    if (cache->slots == NULL)
    {
        cache->slots = calloc(EVAL_CACHE_ENTRIES, sizeof(_Atomic uint64_t));
    }
}

void eval_cache_free(EvalCache *cache)
{
    free((void *)cache->slots);
    cache->slots = NULL;
}

void eval_cache_clear(EvalCache *cache)
{
    if (cache->slots)
    {
        memset((void *)cache->slots, 0, EVAL_CACHE_ENTRIES * sizeof(_Atomic uint64_t));
    }
}

int evaluate(const Position *pos, EvalCache *cache)
{
    // cache may be NULL (uncached callers) or its allocation may have failed — degrade to an uncached eval
    // rather than dereferencing NULL. The branch is perfectly predicted in every normal run.
    _Atomic uint64_t *const slot = (cache && cache->slots) ? &cache->slots[pos->key & (EVAL_CACHE_ENTRIES - 1)] : NULL;
    if (slot)
    {
        const uint64_t entry = atomic_load_explicit(slot, memory_order_relaxed);
        if (entry != 0 && ((entry ^ pos->key) & ~0xFFFFull) == 0)
        {
            return (int16_t)(uint16_t)entry; // hit: the low 16 bits hold the cached eval
        }
    }

    // The incrementally-maintained accumulator (kept in sync by make_move/set_fen) makes this a cheap
    // forward pass. The net is never NULL: engine_new() loads the embedded net before anything searches.
    const int value = nnue_evaluate(&pos->accumulator, pos->color_to_move, nnue_output_bucket(pos));
    if (slot)
    {
        atomic_store_explicit(slot, eval_cache_pack(pos->key, value), memory_order_relaxed);
    }
    return value;
}
