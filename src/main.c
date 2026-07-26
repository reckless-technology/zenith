// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief Program entry point: construct the engine, run, release.
 */
#include "engine.h"
#include "uci.h"
#include <stdio.h>

/** @brief Program entry: build the one Engine (which performs all initialization), run, release. */
int main(int argc, char **argv)
{
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
