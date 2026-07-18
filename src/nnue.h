#pragma once
// NNUE evaluation: loads a Zenith .nnue net and evaluates positions with a quantised SCReLU perspective
// network. The integer forward pass here must stay byte-identical to trainer/features.py::integer_eval
// (verified to 0 cp on a fixed FEN set). v1 recomputes the accumulator from the board each call
// (full refresh); an incremental accumulator is a later speed optimisation.
#include "position.h"
#include <string>

namespace nnue
{
bool load(const std::string &path); // read a .nnue file; returns true on success
bool is_loaded();
int  evaluate(const Position &position); // centipawns, side-to-move POV

// Verification helper: read FENs from stdin (one per line) and print the NNUE eval of each.
int eval_fens_from_stdin(const std::string &net_path);
} // namespace nnue
