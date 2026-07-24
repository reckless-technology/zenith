#pragma once
#include <stdalign.h>
#include <stdint.h>

// One NNUE hidden-layer accumulator per perspective, embedded in Position and maintained incrementally
// by put/remove/move_piece (so make_move and set_fen keep it in sync). Eval reads values[stm] as "own"
// and values[~stm] as "opponent". Kept here (not in nnue.h) so position.h can hold it without a cycle.
enum
{
    NNUE_HIDDEN = 512 // must equal HIDDEN_SIZE in nnue.c / features.py
};

typedef struct NnueAccumulator
{
    _Alignas(32) int16_t values[2][NNUE_HIDDEN];
    // Cached king-input bucket per perspective (WHITE, BLACK), set by nnue_refresh / refresh_perspective.
    // Incremental updates use these; a king move that changes a side's bucket refreshes that perspective.
    int king_bucket[2];
} NnueAccumulator;
