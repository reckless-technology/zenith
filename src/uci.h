// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief The UCI protocol loop and the CLI self-test subcommands.
 */
#pragma once
/** @brief Run the interactive UCI protocol loop, reading commands from stdin until "quit"/EOF. */
void uci_loop(void);
/** @brief Fixed-depth benchmark over a canonical FEN set; prints the deterministic node signature and nps. */
void run_bench(int depth);
/** @brief Run the perft suite (canonical + edge-case catchers + Ethereal 128) against known counts.
 *  @return 0 if every position matches, non-zero on any mismatch. */
int run_perft_suite(void);
/** @brief Differential gate: is_legal_fast / gives_check_fast vs copy-make truth. @return non-zero on failure. */
int run_legal_check(void);
/** @brief Adversarial-input gate over malformed/hostile FEN input. @return non-zero on failure. */
int run_fuzz_check(void);
/** @brief SEE unit test over hand-verified capture positions. @return non-zero on failure. */
int run_see_check(void);
