#pragma once
#include "types.h"
#include <atomic>
#include <cstdint>
#include <vector>

enum Bound : uint8_t
{
    BOUND_NONE  = 0,
    BOUND_UPPER = 1,
    BOUND_LOWER = 2,
    BOUND_EXACT = 3
};

// Unpacked probe result (the search reads this).
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

// Lockless transposition table for Lazy SMP: each 16-byte slot stores {key ^ data, data}. A torn read
// (data and key from different writes) fails the `key ^ data == probeKey` check and is treated as a miss,
// so threads can probe/store concurrently without locks (occasional benign misses on races). Accesses go
// through std::atomic_ref so the u64 loads/stores are well-defined.
class TranspositionTable
{
    struct Slot
    {
        uint64_t key  = 0; // realKey ^ data
        uint64_t data = 0; // packed move|score|eval|depth|bound|gen
    };

    std::vector<Slot> table;
    uint64_t          mask       = 0;
    uint8_t           generation = 0;

  public:
    void resize(size_t megabytes);
    void clear();

    void new_search()
    {
        generation++;
    }

    // probe: returns true on a key hit (with a real entry), filling out.
    bool probe(uint64_t key, TTEntry &out);
    void store(uint64_t key, int score, int eval, int depth, Bound bound, Move move, int ply);
    int  hashfull();
};

extern TranspositionTable TT;

// Mate scores are stored as distance-from-this-node; convert on the way in/out so a mate found deep in
// the tree is scored correctly wherever the entry is reused.
inline int score_to_tt(int score, int ply)
{
    if (score >= VALUE_MATE_IN_MAX)
    {
        return score + ply;
    }
    if (score <= -VALUE_MATE_IN_MAX)
    {
        return score - ply;
    }
    return score;
}

inline int score_from_tt(int score, int ply)
{
    if (score == VALUE_NONE)
    {
        return score;
    }
    if (score >= VALUE_MATE_IN_MAX)
    {
        return score - ply;
    }
    if (score <= -VALUE_MATE_IN_MAX)
    {
        return score + ply;
    }
    return score;
}
