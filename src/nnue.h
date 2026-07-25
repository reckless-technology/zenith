// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief NNUE evaluation: a quantised SCReLU perspective network with an incrementally-maintained accumulator.
 *
 * Position embeds an NnueAccumulator that put/remove/move_piece update, so make_move/set_fen keep it in
 * sync and eval is a cheap forward pass (no per-node full refresh). The integer forward stays byte-identical
 * to the integer_eval reference in trainer/features.py (0 cp gate); `nnuecheck` verifies incremental == refresh.
 */
#pragma once
#include "accumulator.h"
#include "position.h"

extern bool nnue_g_is_loaded; ///< whether a net has been loaded (see nnue_load)

/** @brief Whether a net is currently loaded (guard incremental-update calls with this). */
static inline bool nnue_is_loaded(void)
{
    return nnue_g_is_loaded;
}

/** @brief Load a net from @p path, committing only on a fully-successful read. @return false on failure. */
bool nnue_load(const char *path);

/// @name Evaluation (centipawns, side-to-move POV).
/// @{
/** @brief Standalone eval of @p position: full refresh then forward pass. */
int nnue_evaluate_position(const Position *position);
/** @brief Forward pass from a maintained @p accumulator, from @p stm's POV. */
int nnue_evaluate(const NnueAccumulator *accumulator, Color stm);
/// @}

/// @name Accumulator maintenance (no-ops for callers to guard with nnue_is_loaded()).
/// @{
/** @brief Recompute both perspectives of @p accumulator from @p position. */
void nnue_refresh(NnueAccumulator *accumulator, const Position *position);
/** @brief Recompute a single @p perspective of @p accumulator from @p position. */
void nnue_refresh_perspective(NnueAccumulator *accumulator, const Position *position, Color perspective);
/** @brief Add the feature for a @p color @p type piece appearing on @p square. */
void nnue_add_feature(NnueAccumulator *accumulator, Color color, PieceType type, int square);
/** @brief Remove the feature for a @p color @p type piece leaving @p square. */
void nnue_remove_feature(NnueAccumulator *accumulator, Color color, PieceType type, int square);
/** @brief Update the feature for a @p color @p type piece moving @p from -> @p to. */
void nnue_move_feature(NnueAccumulator *accumulator, Color color, PieceType type, int from, int to);
/**
 * @brief After a king move: if @p side's king-input bucket changed, refresh that perspective.
 *
 * Its whole feature block shifts. Call once per king move (including castling) from make_move.
 */
void nnue_update_king_bucket(NnueAccumulator *accumulator, const Position *position, Color side);
/// @}

/// @name Verification helpers.
/// @{
/** @brief Load @p net_path and print the eval of each FEN read on stdin (the 0 cp gate). */
int nnue_eval_fens_from_stdin(const char *net_path);
/** @brief Load @p net_path and check incremental == refresh over a perft-like walk. */
int nnue_run_self_check(const char *net_path);
/// @}
