#include "tt.h"
#include <bit>
#include <cstring>

TranspositionTable TT;

void TranspositionTable::resize(size_t mb)
{
    size_t bytes = mb * 1024 * 1024;
    size_t n     = bytes / sizeof(TTEntry);
    if (n < 1024)
    {
        n = 1024;
    }
    n = std::bit_floor(n); // power of two so index is a mask
    table.assign(n, TTEntry{});
    mask       = n - 1;
    generation = 0;
}

void TranspositionTable::clear()
{
    std::fill(table.begin(), table.end(), TTEntry{});
    generation = 0;
}

bool TranspositionTable::probe(uint64_t key, TTEntry &out) const
{
    const TTEntry &e = table[key & mask];
    if (e.key == key && e.bound != BOUND_NONE)
    {
        out = e;
        return true;
    }
    return false;
}

void TranspositionTable::store(uint64_t key, int score, int eval, int depth, Bound b, Move m, int ply)
{
    TTEntry &e = table[key & mask];
    // Replace when: empty, this position, from an older search, or a deeper/exact result. Preserve a TT
    // move if the new store has none (a fail-low often lacks a best move).
    if (e.bound != BOUND_NONE && e.key == key && m.is_none())
    {
        m = Move(e.move);
    }
    if (e.bound == BOUND_NONE || e.key == key || e.gen != generation || depth + (b == BOUND_EXACT ? 2 : 0) >= e.depth)
    {
        e.key   = key;
        e.score = score_to_tt(score, ply);
        e.eval  = eval;
        e.move  = m.raw();
        e.depth = int16_t(depth);
        e.bound = b;
        e.gen   = generation;
    }
}

int TranspositionTable::hashfull() const
{
    int used   = 0;
    int sample = table.size() < 1000 ? int(table.size()) : 1000;
    for (int i = 0; i < sample; i++)
    {
        if (table[i].bound != BOUND_NONE && table[i].gen == generation)
        {
            used++;
        }
    }
    return used * 1000 / sample;
}
