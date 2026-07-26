// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief Program entry point: one-time table init, engine construction, then uci_run.
 */
#include "bitboard.h"
#include "engine.h"
#include "uci.h"
#include <stdio.h>

/** @brief Program entry: fill the magic tables, build the one Engine, run, release. */
int main(int argc, char **argv)
{
    init_bitboards(); // the magic sliding-attack tables — the one remaining startup init

    Engine *const engine = engine_new(); // the one engine instance; everything else borrows it
    if (engine == NULL)
    {
        fprintf(stderr, "failed to initialise engine (out of memory)\n");
        return 1;
    }
    const int status = uci_run(engine, argc, argv);
    engine_delete(engine);
    return status;
}
