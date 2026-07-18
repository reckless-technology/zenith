#pragma once
#include "movegen.h"
#include "position.h"
#include "tt.h"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <vector>

struct SearchLimits {
    int64_t time[2] = {0, 0}; // wtime, btime (ms); 0 = not given
    int64_t inc[2] = {0, 0};
    int movestogo = 0;
    int64_t movetime = 0; // fixed ms/move
    int depth = 0;        // fixed depth
    int64_t nodes = 0;    // node cap
    bool infinite = false;
};

void init_search();

class Searcher {
  public:
    std::atomic<bool> stop{false};
    uint64_t nodes = 0;
    int seldepth = 0;
    int64_t moveOverhead = 20;

    // Repetition/50-move context: keys of positions played before the root (from UCI), extended in-tree.
    std::vector<uint64_t> hist;

    // Search the root and return the best move (prints UCI info lines).
    Move go(Position root, const SearchLimits& lim);

  private:
    std::chrono::steady_clock::time_point start;
    int64_t softMs = 0, hardMs = 0, nodeLimit = 0;
    bool useTime = false;

    Move killers[MAX_PLY][2];
    int history[2][64][64];
    Move pvTable[MAX_PLY][MAX_PLY];
    int pvLen[MAX_PLY];
    Move rootBest;

    int64_t elapsed() const;
    bool time_up();
    void set_time(const Position& root, const SearchLimits& lim);

    bool is_draw(const Position& pos) const;
    int negamax(Position& pos, int depth, int alpha, int beta, int ply, bool cutnode);
    int qsearch(Position& pos, int alpha, int beta, int ply);
    void update_pv(int ply, Move m);
};
