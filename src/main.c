// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief Program entry point: startup init order, then dispatch a CLI subcommand or the UCI loop.
 */
#include "bitboard.h"
#include "book.h"
#include "datagen.h"
#include "engine.h"
#include "eval.h"
#include "nnue.h"
#include "position.h"
#include "search.h"
#include "tt.h"
#include "uci.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/** @brief Dispatch a CLI subcommand if given, else run the UCI loop. @return the process exit status. */
static int run(Engine *engine, int argc, char **argv)
{
    if (argc > 1)
    {
        if (!strcmp(argv[1], "bench"))
        {
            return run_bench(engine,
                             argc > 2 ? atoi(argv[2]) : 0); // 0 => the pinned default depth; exit code = pass/fail
        }
        if (!strcmp(argv[1], "perft"))
        {
            return run_perft_suite(); // exit code reflects pass/fail (used by `make check`)
        }
        if (!strcmp(argv[1], "bookcheck"))
        {
            return run_book_check(); // polyglot key vs the 9 official spec vectors
        }
        if (!strcmp(argv[1], "fuzzcheck"))
        {
            return run_fuzz_check(); // malformed-input hardening (meaningful under an ASan/UBSan build)
        }
        if (!strcmp(argv[1], "seecheck"))
        {
            return run_see_check(); // static_exchange_eval vs hand-verified capture positions
        }
        if (!strcmp(argv[1], "legalcheck"))
        {
            return run_legal_check(); // position_is_legal == position_is_legal_slow over a perft-like walk
        }
        if (!strcmp(argv[1], "datagen"))
        {
            return run_datagen(engine, argc - 1, argv + 1);
        }
        if (!strcmp(argv[1], "bullet2text"))
        {
            return run_bullet2text(argc - 1, argv + 1);
        }
        if (!strcmp(argv[1], "nnueeval"))
        {
            if (argc < 3)
            {
                fprintf(stderr, "usage: %s nnueeval <net.nnue>   (reads FENs from stdin)\n", argv[0]);
                return 1;
            }
            return nnue_eval_fens_from_stdin(argv[2]);
        }
        if (!strcmp(argv[1], "nnuecheck"))
        {
            if (argc < 3)
            {
                fprintf(stderr, "usage: %s nnuecheck <net.nnue>   (incremental == refresh gate)\n", argv[0]);
                return 1;
            }
            return nnue_run_self_check(argv[2]);
        }
    }
    uci_loop(engine);
    return 0;
}

/** @brief Program entry: initialise subsystems (order matters), build the one Engine, run, release. */
int main(int argc, char **argv)
{
    init_bitboards(); // the magic sliding-attack tables — the one remaining startup init

    Engine *const engine = engine_new(); // the one engine instance; everything else borrows it
    if (engine == NULL)
    {
        fprintf(stderr, "failed to initialise engine (out of memory)\n");
        return 1;
    }
    const int status = run(engine, argc, argv);
    engine_delete(engine);
    return status;
}
