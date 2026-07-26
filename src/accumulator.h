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

typedef struct NnueNetwork      NnueNetwork;      ///< a loaded quantised net (layout private to nnue.c)
typedef struct NnueRefreshCache NnueRefreshCache; ///< per-thread accumulator refresh cache (see nnue.h)

/**
 * @brief One NNUE hidden-layer accumulator per perspective, embedded in Position.
 *
 * Maintained incrementally by put/remove/move_piece (so make_move and set_fen keep it in sync). Eval reads
 * values[stm] as "own" and values[~stm] as "opponent".
 *
 * The accumulator also carries its bindings: the net it is maintained against (NULL = no net, HCE eval — the
 * incremental updates become no-ops) and an optional per-thread refresh cache used when a king-bucket change
 * forces a perspective rebuild. Both pointers sit in the struct's pre-existing alignment padding, so
 * embedding them costs no Position bytes, and copy-make propagates them to children for free.
 */
typedef struct NnueAccumulator
{
    _Alignas(32) int16_t values[2][NNUE_HIDDEN]; ///< hidden layer per perspective (WHITE, BLACK)
    /// Cached king-input bucket per perspective, set by nnue_refresh / refresh_perspective. Incremental
    /// updates use these; a king move that changes a side's bucket refreshes that perspective.
    int                king_bucket[2];
    const NnueNetwork *net;   ///< the net this accumulator tracks (NULL = none; position_init binds it)
    NnueRefreshCache  *cache; ///< refresh cache for king-bucket rebuilds (NULL = rebuild by full rescan)
} NnueAccumulator;

// Both pointers must land in the tail padding the 32-byte alignment already creates — Position's size (and
// therefore the per-node copy-make cost) must not change: 2*NNUE_HIDDEN*2 bytes of values + 2 ints +
// 2 pointers = 2072 <= 2080 (the struct's padded size without them).
_Static_assert(sizeof(NnueAccumulator) == 2080, "accumulator bindings must fit the existing tail padding");
