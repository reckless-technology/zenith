// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief The Engine aggregate: all cross-thread engine state, owned by the entry point.
 *
 * One Engine instance holds the state every search thread shares (the transposition table today; the search
 * parameters, eval cache, and NNUE network join it as the de-globalization proceeds). main() owns the
 * instance on its stack and passes it explicitly to the UCI loop, bench, and datagen — there is no global
 * engine state, so tests and future multi-instance embeddings cannot alias each other.
 */
#pragma once
#include "accumulator.h"
#include "eval.h"
#include "tt.h"
#include "types.h"
#include <stdatomic.h>

/**
 * @brief Tunable search parameters, exposed as UCI spin options for SPSA tuning.
 *
 * Defaults reproduce the shipped engine exactly, so the bench signature is unchanged. After tuning, the
 * winning values are baked back here. Current defaults are SPSA-tuned twice: pass 1 (800 iterations,
 * +26.5 Elo SPRT) and pass 2 (500 iterations from those values, +11.1 +/- 6.0 Elo SPRT at 8+0.08).
 */
typedef struct SearchParams
{
    int rfp_margin;         ///< reverse-futility margin per depth
    int nmp_divisor;        ///< null-move reduction: +min((eval-beta)/nmp_divisor, 3)
    int lmp_base;           ///< late-move-pruning count: base + depth*depth
    int futility_base;      ///< futility margin base
    int futility_margin;    ///< futility margin per depth
    int see_capture_margin; ///< SEE capture-pruning threshold per depth
    int lmr_base_x100;      ///< LMR base (x100): reduction = lmr_base/100 + ln(d)*ln(m)/(lmr_divisor/100)
    int lmr_divisor_x100;   ///< LMR divisor (x100)
    int singular_margin;    ///< singular-extension beta margin per depth
    int aspiration_delta;   ///< initial aspiration half-window
    int history_max;        ///< history bonus cap (min(depth*depth, history_max))
} SearchParams;

/**
 * @brief Search state shared (read-mostly) by every Lazy-SMP thread of one engine.
 *
 * `is_stop_requested` is the one live flag: the main thread (or a UCI "stop") sets it and every thread
 * exits. Keeping the
 * shared state out of Searcher keeps Searcher copyable, so a thread pool can live in one flat allocation.
 */
typedef struct SearchShared
{
    atomic_bool  is_stop_requested;       ///< set to request that all threads abort their searches
    SearchParams params;                  ///< live tunable parameters (UCI spins)
    int          reductions[MAX_PLY][64]; ///< LMR table, rebuilt whenever an LMR param changes
} SearchShared;

/** @brief Fill @p shared with the default parameters and build its LMR reduction table. */
void search_shared_init(SearchShared *shared);

/**
 * @brief Set a tunable param by UCI option name (e.g. "RfpMargin").
 * @return true if the name matched. Recomputes @p shared's LMR table when an LMR param changes.
 */
bool set_search_param(SearchShared *shared, const char *name, int value);

/** @brief One engine instance: the state shared by all of its search threads. Create with engine_new and
 *  destroy with engine_delete. */
typedef struct Engine
{
    TranspositionTable tt;         ///< the shared lockless transposition table
    SearchShared       search;     ///< stop flag, tunable parameters, LMR table
    EvalCache          eval_cache; ///< shared lockless eval memoisation
    const NnueNetwork *net;        ///< the loaded NNUE net (owned; NULL = HCE evaluation)
} Engine;

enum
{
    ENGINE_DEFAULT_HASH_MB = 64 ///< TT size a fresh engine starts with (the UCI Hash option resizes it)
};

/**
 * @brief Allocate, construct, and fully initialize a new Engine: default search parameters + LMR table,
 * eval cache, and a ENGINE_DEFAULT_HASH_MB transposition table. No net is loaded (HCE) until the caller
 * sets one. @return the ready-to-search engine, or NULL on allocation failure (nothing leaked).
 */
Engine *engine_new(void);

/** @brief Destroy an engine from engine_new: release everything it owns (TT, eval cache, net) and free it.
 *  Safe on NULL. No search thread may still be running against it. */
void engine_delete(Engine *engine);
