#pragma once
#include "types.h"
#include <bit>
#include <cstdint>
#include <vector>

enum Bound : uint8_t
{
    BOUND_NONE  = 0,
    BOUND_UPPER = 1,
    BOUND_LOWER = 2,
    BOUND_EXACT = 3
};

// The 64-bit transposition-table payload as a bit-field struct. Field access compiles to the same shift/mask
// the old hand-packing used, but only for the fields a caller actually reads (no eager unpack), and stores
// compose the whole word in one go via std::bit_cast. All fields use 64-bit base types so the struct is a
// single 8-byte allocation unit (LSB-first on this ABI); signed bit-fields sign-extend on read.
struct TTData
{
    uint64_t move : 16;  // packed Move (0 = none)
    int64_t  score : 16; // score_to_tt-adjusted search score
    int64_t  eval : 16;  // raw static eval (VALUE_NONE if in check)
    int64_t  depth : 8;  // SIGNED [-128,127]: qsearch-style entries (depth <= 0) must not wrap to "deep"
    uint64_t bound : 2;  // Bound
    uint64_t gen : 6;    // generation the entry was written in
};

static_assert(sizeof(TTData) == 8, "TTData must pack into one 64-bit word");

// Lockless transposition table for Lazy SMP: each 16-byte slot stores {key ^ data, data}. A torn read
// (data and key from different writes) fails the `key ^ data == probe key` check and is treated as a miss,
// so threads can probe/store concurrently without locks (occasional benign misses on races). Accesses go
// through std::atomic_ref so the u64 loads/stores are well-defined.
class TranspositionTable
{
    struct Slot
    {
        uint64_t key  = 0; // real key ^ data
        uint64_t data = 0; // std::bit_cast<uint64_t>(TTData)
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

    // probe: returns true on a key hit with a real entry, copying the one-word payload into out.
    bool probe(uint64_t key, TTData &out) const;
    void store(uint64_t key, int score, int eval, int depth, Bound bound, Move move, int ply);
    int  hashfull() const;
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
