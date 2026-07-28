// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief Entry point of the standalone bulletformat-to-text converter (`make datagen` ->
 * ./build/zenith-bullet2text).
 */
#include "datagen.h"

/** @brief Program entry: pure byte-level conversion — no Engine (and no search) is needed. */
int main(int argc, char **argv)
{
    return run_bullet2text(argc, argv);
}
