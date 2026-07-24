#pragma once
#include "movegen.h"
#include "position.h"
#include "tt.h"
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

typedef struct SearchLimits
{
    int64_t time[2]; // wtime, btime (ms); 0 = not given
    int64_t inc[2];
    int     movestogo;
    int64_t movetime; // fixed ms/move
    int     depth;    // fixed depth
    int64_t nodes;    // node cap
    bool    infinite;
    bool    has_time_control; // a clock/movetime token was given (so a 0/negative clock ⇒ move now, not hang)
} SearchLimits;

// The C++ relied on default member initializers; C callers zero the struct through this instead.
static inline void search_limits_init(SearchLimits *limits)
{
    memset(limits, 0, sizeof *limits);
}

void init_search(void);

// Tunable search parameters, exposed as UCI spin options for SPSA tuning. Defaults reproduce the shipped
// engine exactly, so the bench signature is unchanged. After tuning, the winning values are baked back here.
typedef struct SearchParams
{
    // Defaults are SPSA-tuned (800 iterations self-play, +26.5 Elo SPRT vs the pre-tune values).
    int rfp_margin;         // reverse-futility margin per depth
    int nmp_divisor;        // null-move reduction: +min((eval-beta)/nmp_divisor, 3)
    int lmp_base;           // late-move-pruning count: base + depth*depth
    int futility_base;      // futility margin base
    int futility_margin;    // futility margin per depth
    int see_capture_margin; // SEE capture-pruning threshold per depth
    int lmr_base_x100;      // LMR base (x100): reduction = lmr_base/100 + ln(d)*ln(m)/(lmr_divisor/100)
    int lmr_divisor_x100;   // LMR divisor (x100)
    int singular_margin;    // singular-extension beta margin per depth
    int aspiration_delta;   // initial aspiration half-window
    int history_max;        // history bonus cap (min(depth*depth, history_max))
} SearchParams;

extern SearchParams g_params;
// Set a tunable param by UCI option name (e.g. "RfpMargin"); returns true if the name matched. Recomputes
// the LMR table when an LMR param changes.
bool set_search_param(const char *name, int value);

// Shared across all Lazy-SMP search threads: the main thread (or a UCI "stop") sets it and every thread
// exits. A single global keeps Searcher copyable so a thread pool can live in one flat allocation.
extern atomic_bool g_stop;

// Repetition/50-move context capacity: keys of positions played before the root (from UCI) plus the
// in-tree path. 8192 covers any practical game (a UCI move list of thousands of plies) plus MAX_PLY of
// search extension; the C++ used an unbounded std::vector.
enum
{
    SEARCH_HIST_CAP = 8192
};

typedef struct Searcher
{
    uint64_t nodes;
    int      seldepth;
    int64_t  move_overhead;
    bool     silent;     // suppress UCI info lines (datagen / bench batches)
    int      root_score; // score (cp, root stm POV) of the last completed iteration — for datagen labels

    // Repetition/50-move context: keys of positions played before the root (from UCI), extended in-tree.
    uint64_t hist_keys[SEARCH_HIST_CAP];
    int      hist_count;

    // --- internals (the C++ kept these private) ---
    int64_t start_ms;
    int64_t soft_ms, hard_ms, node_limit;
    bool    use_time;
    bool    is_main;

    Move killers[MAX_PLY][2];
    int  history[2][64][64];
    int  correction_history[2][16384]; // [stm][pawn_key] eval correction (game-phase agnostic)
    Move counter_moves[768];           // [prev (piece,to)] -> refutation move
    int  cont_hist[768 * 768];         // [prev (piece,to)][cur (piece,to)] 1-ply continuation history
    Move pv_table[MAX_PLY][MAX_PLY];
    int  pv_len[MAX_PLY];
    Move root_best;
} Searcher;

// Mirrors the C++ default constructor (zeroed cont_hist, move_overhead 20, not silent, empty hist).
// The struct is ~2.4 MB — heap-allocate Searchers (the UCI thread pool is malloc'd).
static inline void searcher_init(Searcher *searcher)
{
    memset(searcher, 0, sizeof *searcher);
    searcher->move_overhead = 20;
}

static inline void searcher_hist_push(Searcher *searcher, uint64_t key)
{
    searcher->hist_keys[searcher->hist_count++] = key;
}

static inline void searcher_hist_pop(Searcher *searcher)
{
    searcher->hist_count--;
}

// Search the root and return the best move. The main thread manages time + prints UCI info; Lazy-SMP
// helper threads (is_main_thread=false) search silently to share TT work and stop when the main does.
Move searcher_go(Searcher *searcher, Position root, const SearchLimits *lim, bool is_main_thread);
