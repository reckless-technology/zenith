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

// Shared across all Lazy-SMP search threads: the main thread (or a UCI "stop") sets it and every thread
// exits. A single global keeps Searcher copyable so a thread pool can live in a std::vector.
extern std::atomic<bool> g_stop;

class Searcher
{
  public:
    uint64_t nodes        = 0;
    int      seldepth     = 0;
    int64_t  moveOverhead = 20;
    bool     silent       = false; // suppress UCI info lines (datagen / bench batches)
    int      rootScore    = 0;     // score (cp, root stm POV) of the last completed iteration — for datagen labels

    Searcher() : contHist(768 * 768, 0)
    {
    }

    // Repetition/50-move context: keys of positions played before the root (from UCI), extended in-tree.
    std::vector<uint64_t> hist;

    // Search the root and return the best move. The main thread manages time + prints UCI info; Lazy-SMP
    // helper threads (isMainThread=false) search silently to share TT work and stop when the main does.
    Move go(Position root, const SearchLimits &lim, bool isMainThread = true);

  private:
    std::chrono::steady_clock::time_point start;
    int64_t                               softMs = 0, hardMs = 0, nodeLimit = 0;
    bool                                  useTime = false;
    bool                                  isMain  = true;

    Move             killers[MAX_PLY][2];
    int              history[2][64][64];
    int              correctionHistory[2][16384]; // [stm][pawnKey] eval correction (game-phase agnostic)
    Move             counterMoves[768];           // [prev (piece,to)] -> refutation move
    std::vector<int> contHist;                    // [prev (piece,to)][cur (piece,to)] 1-ply continuation history
    Move             pvTable[MAX_PLY][MAX_PLY];
    int              pvLen[MAX_PLY];
    Move             rootBest;

    int64_t elapsed() const;
    bool    time_up();
    void    set_time(const Position &root, const SearchLimits &lim);

    bool is_draw(const Position &pos) const;
    int  negamax(Position &pos, int depth, int alpha, int beta, int ply, bool cutnode, Move prevMove,
                 Move excluded = Move::none());
    int  qsearch(Position &pos, int alpha, int beta, int ply);
    void update_pv(int ply, Move m);
};
