#pragma once
#include "types.h"
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef enum
{
    BOUND_NONE  = 0,
    BOUND_UPPER = 1,
    BOUND_LOWER = 2,
    BOUND_EXACT = 3
} Bound;

#define TT_MAX_MB 65536 // must match the advertised `Hash` spin max in uci.c

// The 64-bit transposition-table payload as a bit-field struct. Field access compiles to the same shift/mask
// the old hand-packing used, but only for the fields a caller actually reads (no eager unpack), and stores
// compose the whole word in one go via memcpy (the C spelling of std::bit_cast). All fields use 64-bit base
// types so the struct is a single 8-byte allocation unit (LSB-first on this ABI); signed bit-fields
// sign-extend on read.
typedef struct TTData
{
    uint64_t move : 16;  // packed Move (0 = none)
    int64_t  score : 16; // score_to_tt-adjusted search score
    int64_t  eval : 16;  // raw static eval (VALUE_NONE if in check)
    int64_t  depth : 8;  // SIGNED [-128,127]: qsearch-style entries (depth <= 0) must not wrap to "deep"
    uint64_t bound : 2;  // Bound
    uint64_t gen : 6;    // generation the entry was written in
} TTData;

_Static_assert(sizeof(TTData) == 8, "TTData must pack into one 64-bit word");

static inline uint64_t tt_data_to_u64(TTData data)
{
    uint64_t word;
    memcpy(&word, &data, sizeof word);
    return word;
}

static inline TTData u64_to_tt_data(uint64_t word)
{
    TTData data;
    memcpy(&data, &word, sizeof data);
    return data;
}

// Lockless transposition table for Lazy SMP: each 16-byte slot stores {key ^ data, data}. A torn read
// (data and key from different writes) fails the `key ^ data == probe key` check and is treated as a miss,
// so threads can probe/store concurrently without locks (occasional benign misses on races). The slot words
// are _Atomic so the relaxed u64 loads/stores are well-defined.
typedef struct TTSlot
{
    _Atomic uint64_t key;  // real key ^ data
    _Atomic uint64_t data; // tt_data_to_u64(TTData)
} TTSlot;

typedef struct TranspositionTable
{
    TTSlot  *table;
    size_t   slot_count;
    uint64_t mask;
    uint8_t  generation;
} TranspositionTable;

extern TranspositionTable TT;

void tt_resize(size_t megabytes);
void tt_clear(void);

static inline void tt_new_search(void)
{
    TT.generation++;
}

// probe: returns true on a key hit with a real entry, copying the one-word payload into out.
bool tt_probe(uint64_t key, TTData *out);
void tt_store(uint64_t key, int score, int eval, int depth, Bound bound, Move move, int ply);
int  tt_hashfull(void);

// Mate scores are stored as distance-from-this-node; convert on the way in/out so a mate found deep in
// the tree is scored correctly wherever the entry is reused.
static inline int score_to_tt(int score, int ply)
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

static inline int score_from_tt(int score, int ply)
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
