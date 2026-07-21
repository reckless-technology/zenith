#pragma once
// NNUE evaluation: a quantised SCReLU perspective network. The accumulator is maintained INCREMENTALLY —
// Position embeds an NnueAccumulator that put/remove/move_piece update, so make_move/set_fen keep it in
// sync and eval is a cheap forward pass (no per-node full refresh). The integer forward stays byte-
// identical to trainer/features.py::integer_eval (0 cp gate); `nnuecheck` verifies incremental == refresh.
#include "accumulator.h"
#include "position.h"
#include <string>

namespace nnue
{
extern bool g_loaded;

inline bool is_loaded()
{
    return g_loaded;
}

bool load(const std::string &path);

// Evaluation (centipawns, side-to-move POV).
int evaluate(const Position &position);                      // standalone: full refresh then forward
int evaluate(const NnueAccumulator &accumulator, Color stm); // from a maintained accumulator

// Accumulator maintenance (no-ops for callers to guard with is_loaded()).
void refresh(NnueAccumulator &accumulator, const Position &position); // recompute both perspectives
void refresh_perspective(NnueAccumulator &accumulator, const Position &position, Color perspective);
void add_feature(NnueAccumulator &accumulator, Color colour, PieceType type, int square);
void remove_feature(NnueAccumulator &accumulator, Color colour, PieceType type, int square);
void move_feature(NnueAccumulator &accumulator, Color colour, PieceType type, int from, int to);
// After a king move: if `side`'s king-input bucket changed, refresh that perspective (its whole feature
// block shifts). Call once per king move (including castling) from make_move.
void update_king_bucket(NnueAccumulator &accumulator, const Position &position, Color side);

// Verification helpers.
int eval_fens_from_stdin(const std::string &net_path); // print eval of each FEN on stdin (0 cp gate)
int run_self_check(const std::string &net_path);       // incremental == refresh over a perft-like walk
} // namespace nnue
