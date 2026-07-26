// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief Move generation into a MOVE_NONE-terminated Move buffer: legal and pseudo-legal generators.
 */
#pragma once
#include "position.h"

/**
 * @brief Move-buffer capacity — provably sufficient, so the generators write unchecked.
 *
 * position_set_fen rejects boards with more than 16 pieces per side (no chess position has more), and any
 * single origin square yields at most 27 moves (a queen on an empty board; a promoting pawn 3 targets x 4
 * pieces = 12; a king 8 + 2 castles = 10). So the side to move emits at most 15 x 27 + 10 = 415 moves; one
 * more slot holds the MOVE_NONE terminator. Callers declare `Move moves[MAX_MOVES]`.
 */
#define MAX_MOVES 416

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
