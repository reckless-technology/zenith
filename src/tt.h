#pragma once
#include "types.h"
#include <vector>

enum Bound : uint8_t
{
    BOUND_NONE  = 0,
    BOUND_UPPER = 1,
    BOUND_LOWER = 2,
    BOUND_EXACT = 3
};

struct TTEntry
{
    uint64_t key   = 0;
    int32_t  score = 0;
    int32_t  eval  = 0;
    uint16_t move  = 0;
    int16_t  depth = -1;
    uint8_t  bound = BOUND_NONE;
    uint8_t  gen   = 0;
};

class TranspositionTable
{
    std::vector<TTEntry> table;
    uint64_t             mask       = 0;
    uint8_t              generation = 0;

  public:
    void resize(size_t mb);
    void clear();

    void new_search()
    {
        generation++;
    }

    // probe: returns true on a key hit, filling *tte.
    bool probe(uint64_t key, TTEntry &out) const;
    void store(uint64_t key, int score, int eval, int depth, Bound b, Move m, int ply);
    int  hashfull() const;
};

extern TranspositionTable TT;

// Mate scores are stored as distance-from-this-node; convert on the way in/out so a mate found deep in
// the tree is scored correctly wherever the entry is reused.
inline int score_to_tt(int s, int ply)
{
    if (s >= VALUE_MATE_IN_MAX)
    {
        return s + ply;
    }
    if (s <= -VALUE_MATE_IN_MAX)
    {
        return s - ply;
    }
    return s;
}

inline int score_from_tt(int s, int ply)
{
    if (s == VALUE_NONE)
    {
        return s;
    }
    if (s >= VALUE_MATE_IN_MAX)
    {
        return s - ply;
    }
    if (s <= -VALUE_MATE_IN_MAX)
    {
        return s + ply;
    }
    return s;
}
