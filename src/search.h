// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief The search: the Searcher state, tunable parameters, per-search limits, and searcher_go.
 */
#pragma once
#include "engine.h"
#include "movegen.h"
#include "nnue.h"
#include "position.h"
#include "tt.h"
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

/** @brief Per-search stopping conditions parsed from a UCI `go` command. */
typedef struct SearchLimits
{
    int64_t time[2];          ///< wtime, btime (ms); 0 = not given
    int64_t inc[2];           ///< winc, binc (ms)
    int     movestogo;        ///< moves until the next time control (0 = sudden death / unspecified)
    int64_t movetime;         ///< fixed ms/move
    int     depth;            ///< fixed depth
    int64_t nodes;            ///< node cap
    bool    is_infinite;      ///< search until "stop"
    bool    has_time_control; ///< a clock/movetime token was given (so a 0/negative clock ⇒ move now, not hang)
} SearchLimits;

/** @brief Zero-initialise @p limits (all fields off / no limit). */
static inline void search_limits_init(SearchLimits *limits)
{
    memset(limits, 0, sizeof *limits);
}

/**
 * @brief Static Exchange Evaluation of a capture: net material after the optimal capture sequence on the
 *        move's target square. Exposed for the `seecheck` unit test.
 * @param pos  the position before the capture.
 * @param move the capture move to evaluate.
 * @return the material swing in centipawns (positive = good for the side to move).
 */
int static_exchange_eval(const Position *pos, Move move);

/**
 * @brief Repetition/50-move context capacity.
 *
 * Keys of positions played before the root (from UCI) plus the in-tree path. 8192 covers any practical game
 * (a UCI move list of thousands of plies) plus MAX_PLY of search depth; the UCI layer caps the pre-root
 * replay to leave that headroom.
 */
enum
{
    SEARCH_HIST_CAP = 8192
};

/** @brief One search thread's complete state (~2.4 MB; heap-allocate these). */
typedef struct Searcher
{
    Engine  *engine;        ///< the owning engine instance (shared TT; every pool thread points at the same one)
    uint64_t nodes;         ///< nodes searched this search
    int      seldepth;      ///< greatest ply reached (selective depth)
    int64_t  move_overhead; ///< ms subtracted from the clock to cover I/O latency
    bool     is_silent;     ///< suppress UCI info lines (datagen / bench batches)
    int      root_score;    ///< score (cp, root stm POV) of the last completed iteration — for datagen labels

    /// Repetition/50-move context: keys of positions played before the root (from UCI), extended in-tree.
    uint64_t hist_keys[SEARCH_HIST_CAP]; ///< pre-root + in-tree position keys
    int      hist_count;                 ///< number of valid entries in @ref hist_keys

    // --- internals ---
    int64_t start_ms;                     ///< search start time (platform_now_ms)
    int64_t soft_ms, hard_ms, node_limit; ///< soft/hard time budgets and node cap
    bool    is_time_limited;              ///< whether a time budget applies
    bool    is_main;                      ///< the main (time-managing, printing) thread vs a Lazy-SMP helper

    Move killers[MAX_PLY][2];          ///< two killer moves per ply
    int  history[2][64][64];           ///< butterfly history [stm][from][to]
    int  correction_history[2][16384]; ///< [stm][pawn_key] eval correction (game-phase agnostic)
    Move counter_moves[768];           ///< [prev (piece,to)] -> refutation move
    int  cont_hist[768 * 768];         ///< [prev (piece,to)][cur (piece,to)] 1-ply continuation history
    Move pv_table[MAX_PLY][MAX_PLY];   ///< triangular principal-variation table
    int  pv_len[MAX_PLY];              ///< PV length per ply
    Move root_best;                    ///< best move at the root of the current search

    /// This thread's NNUE refresh cache; searcher_go binds it into the root position, and copy-make hands
    /// it down to every node this thread searches. Zeroed by searcher_init (entries then miss on first use).
    NnueRefreshCache refresh_cache;
} Searcher;

/**
 * @brief Zero @p searcher and set defaults (move_overhead 20, not silent, empty history).
 *
 * The struct is ~2.4 MB — heap-allocate Searchers (the UCI thread pool is malloc'd).
 */
static inline void searcher_init(Searcher *searcher, Engine *engine)
{
    memset(searcher, 0, sizeof *searcher);
    searcher->engine        = engine;
    searcher->move_overhead = 20;
}

/** @brief Push @p key onto the repetition-history stack (around a recursive search call). */
static inline void searcher_hist_push(Searcher *searcher, uint64_t key)
{
    searcher->hist_keys[searcher->hist_count++] = key;
}

/** @brief Pop the last key off the repetition-history stack. */
static inline void searcher_hist_pop(Searcher *searcher)
{
    searcher->hist_count--;
}

/**
 * @brief Search @p root and return the best move.
 * @param searcher this thread's search state.
 * @param root the root position to search (taken by value).
 * @param lim the stopping conditions for this search.
 * @param is_main_thread the main thread manages time + prints UCI info; Lazy-SMP helpers
 *        (is_main_thread=false) search silently to share TT work and stop when the main does.
 */
Move searcher_go(Searcher *searcher, Position root, const SearchLimits *lim, bool is_main_thread);
