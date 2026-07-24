// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief Polyglot opening book: loading, probing, and the Polyglot Zobrist key.
 *
 * Standard .bin format (big-endian {key u64, move u16, weight u16, learn u32} entries sorted by key).
 * Probing computes the position's Polyglot Zobrist key (its own fixed random array — unrelated to the
 * engine's Zobrist), binary-searches the book, and picks among the entries weighted-randomly for variety.
 * Off by default (UCI OwnBook); testing always runs bookless.
 */
#pragma once
#include "position.h"
#include <stdbool.h>

/** @brief Load a .bin book fully into memory (replaces any previous book). @return false on failure. */
bool book_load(const char *path);
/** @brief Whether a non-empty book is currently loaded. */
bool book_is_loaded(void);
/** @brief Pick a book move for @p pos. @return a verified-legal move, or MOVE_NONE on miss / no book. */
Move book_probe(const Position *pos);

/** @brief The position's Polyglot Zobrist key. Exposed for the bookcheck gate. */
uint64_t polyglot_key(const Position *pos);

/** @brief Validate polyglot_key() against the 9 official spec test vectors. @return non-zero on failure. */
int run_book_check(void);
