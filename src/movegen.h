// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief Move generation into a fixed MoveList: legal and pseudo-legal generators.
 */
#pragma once
#include "position.h"

/**
 * @brief MoveList capacity.
 *
 * A legal chess position has at most 218 moves; 256 covers that with margin. Illegal positions (reachable
 * only via a hand-crafted FEN — e.g. a board full of queens) can exceed it, so movelist_add caps at capacity
 * and drops the overflow rather than writing past the array. Legal positions never hit the cap, so search
 * behaviour (and the bench signature) is unchanged.
 */
#define MOVELIST_CAP 256

/** @brief A fixed-capacity list of generated moves. */
typedef struct MoveList
{
    Move moves[MOVELIST_CAP]; ///< generated moves (up to MOVELIST_CAP)
    int  count;               ///< number of valid entries in @ref moves
} MoveList;

/** @brief Append @p move to @p list, silently dropping it if the list is already at capacity. */
static inline void movelist_add(MoveList *list, Move move)
{
    if (list->count < MOVELIST_CAP)
    {
        list->moves[list->count++] = move;
    }
}

/**
 * @brief Generate fully legal moves from @p pos into @p list (which is reset).
 *
 * With noisy_only set, only captures + promotions are generated (for quiescence).
 */
void generate_legal(const Position *pos, MoveList *list, bool noisy_only);

/**
 * @brief Generate pseudo-legal moves from @p pos into @p list, which is reset (castling is emitted fully legal).
 *
 * The search makes each move once and skips those that leave the mover's king in check — avoiding
 * generate_legal's extra copy-make per move. With noisy_only set, only captures + promotions are generated.
 */
void generate_pseudo(const Position *pos, MoveList *list, bool noisy_only);
