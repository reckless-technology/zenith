// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief The evaluation seam: NNUE when a net is loaded, else the PeSTO tapered HCE.
 */
#pragma once
#include "position.h"

/** @brief Initialise the evaluation tables and cache. Call once at startup. */
void init_eval(void);
/**
 * @brief Static evaluation from the side-to-move's perspective (centipawns).
 *
 * Results are memoised in a small shared lockless eval cache keyed by the Zobrist key (biggest win:
 * qsearch stand-pat re-evaluations).
 */
int evaluate(const Position *pos);
/**
 * @brief Empty the eval cache.
 *
 * Must be called when evaluations change meaning: a new net is loaded (EvalFile) or the user asks for a
 * full reset (Clear Hash).
 */
void eval_cache_clear(void);
