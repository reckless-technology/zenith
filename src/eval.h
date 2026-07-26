// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief The evaluation seam: NNUE when a net is loaded, else the PeSTO tapered HCE.
 */
#pragma once
#include "position.h"
#include <stdatomic.h>

/**
 * @brief A shared lockless eval cache: 2^20 single-u64 entries (8 MB).
 *
 * Each entry packs the key's high 48 bits with the 16-bit eval; the slot index uses the key's low 20 bits,
 * so a verified hit implies ALL 64 key bits match — false hits are impossible, a collision only evicts.
 * Single-word relaxed atomics cannot tear, so one cache is safely shared by all Lazy-SMP threads.
 */
typedef struct EvalCache
{
    _Atomic uint64_t *slots; ///< calloc'd table (NULL = allocation failed; evaluate degrades to uncached)
} EvalCache;

/** @brief Allocate @p cache's zeroed table (idempotent; a failed allocation leaves it NULL = uncached). */
void eval_cache_init(EvalCache *cache);

/** @brief Initialise the evaluation PST tables. Call once at startup. */
void init_eval(void);
/**
 * @brief Static evaluation from the side-to-move's perspective (centipawns).
 *
 * Results are memoised in a small shared lockless eval cache keyed by the Zobrist key (biggest win:
 * qsearch stand-pat re-evaluations).
 */
int evaluate(const Position *pos, EvalCache *cache);
/**
 * @brief Empty @p cache.
 *
 * Must be called when evaluations change meaning: a new net is loaded (EvalFile) or the user asks for a
 * full reset (Clear Hash).
 */
void eval_cache_clear(EvalCache *cache);
