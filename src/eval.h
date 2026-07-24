#pragma once
#include "position.h"

void init_eval(void);
// Static evaluation from the side-to-move's perspective (centipawns). Results are memoised in a small
// shared lockless eval cache keyed by the Zobrist key (biggest win: qsearch stand-pat re-evaluations).
int evaluate(const Position *pos);
// Empty the eval cache. Must be called when evaluations change meaning: a new net is loaded (EvalFile)
// or the user asks for a full reset (Clear Hash).
void eval_cache_clear(void);
