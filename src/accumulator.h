// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief The NNUE hidden-layer accumulator embedded in Position.
 *
 * Kept in its own header (not nnue.h) so position.h can hold it without an include cycle.
 */
#pragma once
#include <stdalign.h>
#include <stdint.h>

enum
{
    NNUE_HIDDEN = 512 ///< hidden-layer width; must equal HIDDEN_SIZE in nnue.c / features.py
};

/**
 * @brief One NNUE hidden-layer accumulator per perspective, embedded in Position.
 *
 * Maintained incrementally by put/remove/move_piece (so make_move and set_fen keep it in sync). Eval reads
 * values[stm] as "own" and values[~stm] as "opponent".
 */
typedef struct NnueAccumulator
{
    _Alignas(32) int16_t values[2][NNUE_HIDDEN]; ///< hidden layer per perspective (WHITE, BLACK)
    /// Cached king-input bucket per perspective, set by nnue_refresh / refresh_perspective. Incremental
    /// updates use these; a king move that changes a side's bucket refreshes that perspective.
    int king_bucket[2];
} NnueAccumulator;
