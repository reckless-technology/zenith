#pragma once
#include "position.h"

typedef struct MoveList
{
    Move moves[256];
    int  count;
} MoveList;

static inline void movelist_add(MoveList *list, Move move)
{
    list->moves[list->count++] = move;
}

// Fully legal moves. noisy_only restricts to captures + promotions (for quiescence).
void generate_legal(const Position *pos, MoveList *list, bool noisy_only);

// Pseudo-legal moves (castling already fully legal). The search makes each move once and skips those that
// leave the mover's king in check — avoiding generate_legal's extra copy-make per move.
void generate_pseudo(const Position *pos, MoveList *list, bool noisy_only);
