// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief NNUE evaluation: a quantised SCReLU perspective network with an incrementally-maintained accumulator.
 *
 * Position embeds an NnueAccumulator that add_piece/remove_piece/move_piece update, so make_move/set_fen keep it in
 * sync and eval is a cheap forward pass (no per-node full refresh). The accumulator carries the net it
 * tracks (NULL = HCE) — there is no global network. The integer forward stays byte-identical to the
 * integer_eval reference in trainer/features.py (0 cp gate); `nnuecheck` verifies incremental == refresh.
 */
#pragma once
#include "accumulator.h"
#include "position.h"
#include <stddef.h>

enum
{
    NNUE_KING_BUCKETS   = 8, ///< king-input buckets (4 files a-d, after mirroring, x 2 board-halves)
    NNUE_OUTPUT_BUCKETS = 8  ///< material output heads, selected by (piece_count - 2) / 4; must match features.py
};

/** @brief The output head for @p position: total piece count (kings included) mapped by (count - 2) / 4.
 *  set_fen guarantees 2..32 pieces, so the result is always in [0, NNUE_OUTPUT_BUCKETS). */
static inline int nnue_output_bucket(const Position *position)
{
    return (popcount(position->pieces[NO_PIECE]) - 2) / 4;
}

/** @brief One refresh-cache slot: the last accumulator built for a (perspective, king bucket) and its board. */
typedef struct NnueRefreshCacheEntry
{
    _Alignas(32) int16_t values[NNUE_HIDDEN]; ///< cached accumulator half for this (perspective, bucket)
    Bitboard           colors[2];             ///< board occupancy per color when `values` was built
    Bitboard           pieces[NUM_PIECES];    ///< board occupancy per piece type when `values` was built
    const NnueNetwork *net;                   ///< the net `values` belongs to (NULL / different net = miss)
} NnueRefreshCacheEntry;

/**
 * @brief The accumulator refresh cache ("finny tables"): one entry per (perspective, king bucket).
 *
 * A king move that switches a perspective to bucket B rebuilds by applying only the piece diffs versus that
 * bucket's cached board — a handful of column ops instead of a full 32-piece rescan. Always correct (the
 * diffs are exact); an entry built for a different net is simply a miss, which replaces the old global
 * net-generation guard. Each search thread owns one (embedded in Searcher) and binds it into its root
 * position, so Lazy-SMP threads never share a cache.
 */
typedef struct NnueRefreshCache
{
    NnueRefreshCacheEntry entries[2][2][NNUE_KING_BUCKETS]; ///< [perspective][mirrored][bucket]
} NnueRefreshCache;

enum
{
    NNUE_REFUSAL_MAX = 320 ///< room for the whole refusal line: the source's name plus the longest reason
};

/**
 * @brief Where a caller receives the refusal a loader printed, so it can repeat it on another stream.
 *
 * The loader itself writes only to stderr — `nnueeval` and `nnuecheck` write their results to stdout, and a
 * diagnostic mixed into that stream would corrupt the 0 cp verification gate's input. A caller that *can*
 * safely talk on stdout (the UCI layer, whose stdout is a line protocol) passes one of these and prints
 * `message` as an `info string`, so a GUI sees the reason a match harness reads on stderr. NULL asks for
 * nothing and changes no output. Passed explicitly rather than kept in file-scope state: the engine has no
 * globals, and two threads may load nets at once.
 */
typedef struct NnueRefusal
{
    char message[NNUE_REFUSAL_MAX]; ///< the refusal exactly as it went to stderr, or "" if nothing was refused
} NnueRefusal;

/**
 * @brief Load a net in the shared `RNNUE1` container from @p path, committing only on a fully-successful
 * read and a fully-successful validation.
 *
 * A net that fails any check is **refused**: the reason goes to stderr naming the field, the file's value
 * and this engine's, and nothing falls back to another net. See nnue.c for the two checks and why a silent
 * fallback is the one behaviour this container exists to prevent.
 * @param refusal optional (NULL to ignore): receives the refusal line verbatim, for a caller that must
 * also say it somewhere else.
 * @return a heap-allocated net (release with nnue_free), or NULL if the net was refused.
 */
const NnueNetwork *nnue_load(const char *path, NnueRefusal *refusal);
/**
 * @brief Load the network embedded in the binary at build time (tools/embed_net.py -> the
 * embedded_network_data array below). This is the engine's default, so a bare binary is always full NNUE
 * strength with no external files — it is never substituted for a net that was asked for and refused.
 * @param refusal optional (NULL to ignore): receives the refusal line verbatim, as nnue_load's does.
 * @return a heap-allocated net (release with nnue_free); NULL only on allocation failure.
 */
const NnueNetwork *nnue_load_embedded(NnueRefusal *refusal);
/** @brief Release a net returned by nnue_load/nnue_load_embedded (safe on NULL). No position may still
 *  reference it. */
void nnue_free(const NnueNetwork *net);

/// The shipped .nnue file, generated into a C array at build time (build/embedded_net.c).
extern const unsigned char embedded_network_data[];
extern const size_t        embedded_network_size;

/// @name Evaluation (centipawns, side-to-move POV).
/// @{
/** @brief Standalone eval of @p position (which must have a bound net): full refresh then forward pass. */
int nnue_evaluate_position(const Position *position);
/** @brief Forward pass from a maintained @p accumulator (reads its bound net), from @p stm's POV, using
 *  the given material @p output_bucket (see nnue_output_bucket). */
int nnue_evaluate(const NnueAccumulator *accumulator, Color stm, int output_bucket);
/// @}

/// @name Accumulator maintenance (no-ops without a bound net — callers guard on accumulator->net).
/// @{
/** @brief Recompute both perspectives of @p accumulator from @p position. */
void nnue_refresh(NnueAccumulator *accumulator, const Position *position);
/** @brief Recompute a single @p perspective of @p accumulator from @p position. */
void nnue_refresh_perspective(NnueAccumulator *accumulator, const Position *position, Color perspective);
/** @brief Add the feature for a @p color @p type piece appearing on @p square. */
void nnue_add_feature(NnueAccumulator *accumulator, Color color, Piece type, int square);
/** @brief Remove the feature for a @p color @p type piece leaving @p square. */
void nnue_remove_feature(NnueAccumulator *accumulator, Color color, Piece type, int square);
/** @brief Update the feature for a @p color @p type piece moving @p from -> @p to. */
void nnue_move_feature(NnueAccumulator *accumulator, Color color, Piece type, int from, int to);
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
