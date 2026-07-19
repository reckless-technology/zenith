#pragma once
#include <cstdint>

// One NNUE hidden-layer accumulator per perspective, embedded in Position and maintained incrementally
// by put/remove/move_piece (so make_move and set_fen keep it in sync). Eval reads values[stm] as "own"
// and values[~stm] as "opponent". Kept here (not in nnue.h) so position.h can hold it without a cycle.
constexpr int NNUE_HIDDEN = 512; // must equal HIDDEN_SIZE in nnue.cpp / features.py

struct alignas(32) NnueAccumulator
{
    int16_t values[2][NNUE_HIDDEN];
};
