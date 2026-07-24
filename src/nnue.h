#pragma once
// NNUE evaluation: a quantised SCReLU perspective network. The accumulator is maintained INCREMENTALLY —
// Position embeds an NnueAccumulator that put/remove/move_piece update, so make_move/set_fen keep it in
// sync and eval is a cheap forward pass (no per-node full refresh). The integer forward stays byte-
// identical to trainer/features.py::integer_eval (0 cp gate); `nnuecheck` verifies incremental == refresh.
#include "accumulator.h"
#include "position.h"

extern bool nnue_g_loaded;

static inline bool nnue_is_loaded(void)
{
    return nnue_g_loaded;
}

bool nnue_load(const char *path);

// Evaluation (centipawns, side-to-move POV).
int nnue_evaluate_position(const Position *position);             // standalone: full refresh then forward
int nnue_evaluate(const NnueAccumulator *accumulator, Color stm); // from a maintained accumulator

// Accumulator maintenance (no-ops for callers to guard with nnue_is_loaded()).
void nnue_refresh(NnueAccumulator *accumulator, const Position *position); // recompute both perspectives
void nnue_refresh_perspective(NnueAccumulator *accumulator, const Position *position, Color perspective);
void nnue_add_feature(NnueAccumulator *accumulator, Color color, PieceType type, int square);
void nnue_remove_feature(NnueAccumulator *accumulator, Color color, PieceType type, int square);
void nnue_move_feature(NnueAccumulator *accumulator, Color color, PieceType type, int from, int to);
// After a king move: if `side`'s king-input bucket changed, refresh that perspective (its whole feature
// block shifts). Call once per king move (including castling) from make_move.
void nnue_update_king_bucket(NnueAccumulator *accumulator, const Position *position, Color side);

// Verification helpers.
int nnue_eval_fens_from_stdin(const char *net_path); // print eval of each FEN on stdin (0 cp gate)
int nnue_run_self_check(const char *net_path);       // incremental == refresh over a perft-like walk
