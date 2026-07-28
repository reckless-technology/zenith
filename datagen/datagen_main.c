// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief Entry point of the standalone self-play generator (`make datagen` -> ./build/zenith-datagen).
 */
#include "datagen.h"
#include "engine.h"
#include <stdio.h>

/** @brief Program entry: build the one Engine (the generator runs real searches), generate, release. */
int main(int argc, char **argv)
{
    Engine *const engine = engine_new(); // the one engine instance; everything else borrows it
    if (engine == NULL)
    {
        fprintf(stderr, "failed to initialize engine (out of memory)\n");
        return 1;
    }
    const int status = run_datagen(engine, argc, argv);
    engine_delete(engine);
    return status;
}
