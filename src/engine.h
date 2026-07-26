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
#include "tt.h"
#include "types.h"
#include <stdatomic.h>

/**
 * @brief Tunable search parameters, exposed as UCI spin options for SPSA tuning.
 *
 * Defaults reproduce the shipped engine exactly, so the bench signature is unchanged. After tuning, the
 * winning values are baked back here. Current defaults are SPSA-tuned (800 iterations self-play, +26.5 Elo
 * SPRT vs the pre-tune values).
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
 * `stop` is the one live flag: the main thread (or a UCI "stop") sets it and every thread exits. Keeping the
 * shared state out of Searcher keeps Searcher copyable, so a thread pool can live in one flat allocation.
 */
typedef struct SearchShared
{
    atomic_bool  stop;                    ///< set to abort all threads' searches
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

/** @brief One engine instance: the state shared by all of its search threads. Zero-initialize, then size the
 *  TT with tt_resize and call search_shared_init before searching. */
typedef struct Engine
{
    TranspositionTable tt;     ///< the shared lockless transposition table
    SearchShared       search; ///< stop flag, tunable parameters, LMR table
} Engine;
