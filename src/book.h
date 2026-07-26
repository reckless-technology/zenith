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

typedef struct BookEntry BookEntry; ///< decoded Polyglot entry (layout private to book.c)

/** @brief An opening book instance: the decoded entries plus the pick PRNG. Zero-initialize for "no book". */
typedef struct Book
{
    BookEntry *entries;   ///< decoded, key-sorted entries (owned; NULL when no book is loaded)
    size_t     count;     ///< number of entries
    uint64_t   rng_state; ///< xorshift64* state for the weighted pick (lazily seeded from the clock)
} Book;

/** @brief Load a .bin book fully into @p book (replacing any previous contents). @return false on failure. */
bool book_load(Book *book, const char *path);
/** @brief Whether @p book holds a non-empty book. */
bool book_is_loaded(const Book *book);
/** @brief Pick a book move for @p pos (advances @p book's pick PRNG).
 *  @return a verified-legal move, or MOVE_NONE on miss / no book. */
Move book_probe(Book *book, const Position *pos);
/** @brief Release @p book's entries (safe on a zero-initialized or already-freed book). */
void book_free(Book *book);

/** @brief The position's Polyglot Zobrist key. Exposed for the bookcheck gate. */
uint64_t polyglot_key(const Position *pos);

/** @brief Validate polyglot_key() against the 9 official spec test vectors. @return non-zero on failure. */
int run_book_check(void);
