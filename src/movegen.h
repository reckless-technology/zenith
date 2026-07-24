// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
#pragma once
#include "position.h"

// A legal chess position has at most 218 moves; 256 covers that with margin. Illegal positions (reachable
// only via a hand-crafted FEN — e.g. a board full of queens) can exceed it, so movelist_add caps at capacity
// and drops the overflow rather than writing past the array. Legal positions never hit the cap, so search
// behaviour (and the bench signature) is unchanged.
#define MOVELIST_CAP 256

typedef struct MoveList
{
    Move moves[MOVELIST_CAP];
    int  count;
} MoveList;

static inline void movelist_add(MoveList *list, Move move)
{
    if (list->count < MOVELIST_CAP)
    {
        list->moves[list->count++] = move;
    }
}

// Fully legal moves. noisy_only restricts to captures + promotions (for quiescence).
void generate_legal(const Position *pos, MoveList *list, bool noisy_only);

// Pseudo-legal moves (castling already fully legal). The search makes each move once and skips those that
// leave the mover's king in check — avoiding generate_legal's extra copy-make per move.
void generate_pseudo(const Position *pos, MoveList *list, bool noisy_only);
