#pragma once
#include "position.h"

void init_eval();
// Static evaluation from the side-to-move's perspective (centipawns). The single call site the NNUE
// evaluator will later replace.
int evaluate(const Position &pos);
