#pragma once
// Polyglot opening book: standard .bin format (big-endian {key u64, move u16, weight u16, learn u32}
// entries sorted by key). Probing computes the position's Polyglot Zobrist key (its own fixed random
// array — unrelated to the engine's Zobrist), binary-searches the book, and picks among the entries
// weighted-randomly for variety. Off by default (UCI OwnBook); testing always runs bookless.
#include "position.h"
#include <stdbool.h>

bool book_load(const char *path); // load a .bin book fully into memory (replaces any previous book)
bool book_is_loaded(void);
Move book_probe(const Position *pos); // MOVE_NONE on miss / no book; result is verified legal

uint64_t polyglot_key(const Position *pos); // exposed for the bookcheck gate

int run_book_check(void); // validate polyglot_key against the 9 official spec test vectors
