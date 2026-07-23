#pragma once
#include "movegen.h"
#include "position.h"
#include "tt.h"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <vector>

struct SearchLimits
{
    int64_t time[2]   = {0, 0}; // wtime, btime (ms); 0 = not given
    int64_t inc[2]    = {0, 0};
    int     movestogo = 0;
    int64_t movetime  = 0; // fixed ms/move
    int     depth     = 0; // fixed depth
    int64_t nodes     = 0; // node cap
    bool    infinite  = false;
};

void init_search();

// Tunable search parameters, exposed as UCI spin options for SPSA tuning. Defaults reproduce the shipped
// engine exactly, so the bench signature is unchanged. After tuning, the winning values are baked back here.
struct SearchParams
{
    // Defaults are SPSA-tuned (800 iterations self-play, +26.5 Elo SPRT vs the pre-tune values).
    int rfp_margin         = 62;  // reverse-futility margin per depth
    int nmp_divisor        = 202; // null-move reduction: +min((eval-beta)/nmp_divisor, 3)
    int lmp_base           = 4;   // late-move-pruning count: base + depth*depth
    int futility_base      = 102; // futility margin base
    int futility_margin    = 96;  // futility margin per depth
    int see_capture_margin = 102; // SEE capture-pruning threshold per depth
    int lmr_base_x100      = 86;  // LMR base (x100): reduction = lmr_base/100 + ln(d)*ln(m)/(lmr_divisor/100)
    int lmr_divisor_x100   = 229; // LMR divisor (x100)
    int singular_margin    = 3;   // singular-extension beta margin per depth
    int aspiration_delta   = 21;  // initial aspiration half-window
    int history_max        = 418; // history bonus cap (min(depth*depth, history_max))
};

extern SearchParams g_params;
// Set a tunable param by UCI option name (e.g. "RfpMargin"); returns true if the name matched. Recomputes
// the LMR table when an LMR param changes.
bool set_search_param(const std::string &name, int value);

// Shared across all Lazy-SMP search threads: the main thread (or a UCI "stop") sets it and every thread
// exits. A single global keeps Searcher copyable so a thread pool can live in a std::vector.
extern std::atomic<bool> g_stop;

class Searcher
{
  public:
    uint64_t nodes         = 0;
    int      seldepth      = 0;
    int64_t  move_overhead = 20;
    bool     silent        = false; // suppress UCI info lines (datagen / bench batches)
    int      root_score    = 0;     // score (cp, root stm POV) of the last completed iteration — for datagen labels

    Searcher() : cont_hist(768 * 768, 0)
    {
    }

    // Repetition/50-move context: keys of positions played before the root (from UCI), extended in-tree.
    std::vector<uint64_t> hist;

    // Search the root and return the best move. The main thread manages time + prints UCI info; Lazy-SMP
    // helper threads (is_main_thread=false) search silently to share TT work and stop when the main does.
    Move go(Position root, const SearchLimits &lim, bool is_main_thread = true);

  private:
    std::chrono::steady_clock::time_point start;
    int64_t                               soft_ms = 0, hard_ms = 0, node_limit = 0;
    bool                                  use_time = false;
    bool                                  is_main  = true;

    Move             killers[MAX_PLY][2];
    int              history[2][64][64];
    int              correction_history[2][16384]; // [stm][pawn_key] eval correction (game-phase agnostic)
    Move             counter_moves[768];           // [prev (piece,to)] -> refutation move
    std::vector<int> cont_hist;                    // [prev (piece,to)][cur (piece,to)] 1-ply continuation history
    Move             pv_table[MAX_PLY][MAX_PLY];
    int              pv_len[MAX_PLY];
    Move             root_best;

    int64_t elapsed() const;
    bool    time_up();
    void    set_time(const Position &root, const SearchLimits &lim);

    bool is_draw(const Position &pos) const;
    int  negamax(Position &pos, int depth, int alpha, int beta, int ply, bool cutnode, Move prev_move,
                 Move excluded = Move::none());
    int  qsearch(Position &pos, int alpha, int beta, int ply);
    void update_pv(int ply, Move m);
};
