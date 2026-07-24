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
/** @brief Run the perft suite (canonical 5 + the Ethereal 128-position suite) against known counts. */
void run_perft_suite(void);
/** @brief Differential gate: is_legal_fast / gives_check_fast vs copy-make truth. @return non-zero on failure. */
int run_legal_check(void);
/** @brief Adversarial-input gate over malformed/hostile FEN input. @return non-zero on failure. */
int run_fuzz_check(void);
