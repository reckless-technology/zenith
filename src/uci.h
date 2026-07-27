// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief The UCI protocol loop and the CLI self-test subcommands.
 */
#pragma once
#include "engine.h"
/** @brief Run the interactive UCI protocol loop, reading commands from stdin until "quit"/EOF. */
void uci_loop(Engine *engine);
/** @brief The process entry body: dispatch a CLI subcommand (bench/perft/gates/datagen/...) if one was
 *  given, else run the UCI loop. @return the process exit status. */
int uci_run(Engine *engine, int argc, char **argv);
/** @brief Fixed-depth benchmark over a canonical FEN set; a [PASS]/[FAIL] node-signature gate at the pinned
 *  depth (informational at any other depth). @return 0 if every checked position matched, non-zero on mismatch. */
int run_bench(Engine *engine, int depth);
/** @brief Run the perft suite (canonical + edge-case catchers + Ethereal 128) against known counts.
 *  @return 0 if every position matches, non-zero on any mismatch. */
int run_perft_suite(void);
/** @brief Differential gate: position_is_move_legal / gives_check_fast vs copy-make truth. @return non-zero on failure.
 */
int run_legal_check(void);
/** @brief Adversarial-input gate over malformed/hostile FEN input. @return non-zero on failure. */
int run_fuzz_check(void);
/** @brief SEE unit test over hand-verified capture positions. @return non-zero on failure. */
int run_see_check(void);
