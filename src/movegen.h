// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief Move generation into a MOVE_NONE-terminated Move buffer: legal and pseudo-legal generators.
 */
#pragma once
#include "position.h"

/**
 * @brief Move-buffer capacity.
 *
 * A legal chess position has at most 218 moves; 256 covers that with margin and leaves room for the
 * MOVE_NONE terminator. Illegal positions (reachable only via a hand-crafted FEN — e.g. a board full of
 * queens) can exceed it, so the generators stop at MAX_MOVES-1 and drop the overflow rather than writing
 * past the array. Legal positions never hit the cap, so search behaviour (and the bench signature) is
 * unchanged. Callers declare `Move moves[MAX_MOVES]`.
 */
#define MAX_MOVES 256

/**
 * @brief Generate fully legal moves from @p pos into @p moves (a MAX_MOVES buffer), MOVE_NONE-terminated.
 * @return the number of moves generated (so callers needing a length avoid re-scanning to the terminator).
 *
 * With is_noisy_only set, only captures + promotions are generated (for quiescence).
 */
int generate_legal(const Position *pos, Move *moves, bool is_noisy_only);

/**
 * @brief Generate pseudo-legal moves from @p pos into @p moves (a MAX_MOVES buffer), MOVE_NONE-terminated.
 * @return the number of moves generated.
 *
 * Castling is emitted fully legal. The search makes each move once and skips those that leave the mover's
 * king in check — avoiding generate_legal's extra copy-make per move. With is_noisy_only set, only
 * captures + promotions are generated.
 */
int generate_pseudo(const Position *pos, Move *moves, bool is_noisy_only);
