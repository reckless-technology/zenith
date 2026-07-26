// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief The Engine aggregate: all cross-thread engine state, owned by the entry point.
 *
 * One Engine instance holds the state every search thread shares (the transposition table today; the search
 * parameters, eval cache, and NNUE network join it as the de-globalization proceeds). main() owns the
 * instance on its stack and passes it explicitly to the UCI loop, bench, and datagen — there is no global
 * engine state, so tests and future multi-instance embeddings cannot alias each other.
 */
#pragma once
#include "tt.h"

/** @brief One engine instance: the state shared by all of its search threads. Zero-initialize, then size the
 *  TT with tt_resize before searching. */
typedef struct Engine
{
    TranspositionTable tt; ///< the shared lockless transposition table
} Engine;
