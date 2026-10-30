// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief Self-play data generation and bulletformat conversion for NNUE training.
 *
 * Two sources: `run_datagen` generates Zenith's own self-play data, and `run_bullet2text` converts an
 * external bulletformat dataset. The nets up to `zenith-cap2` trained on the latter — the public
 * PlentyChess dataset, by Yoshie2000 — which is no longer used; see NNUE_TRAINING.md. No engine code
 * is shared with any other engine.
 */
#pragma once
#include "engine.h"
/**
 * @brief Self-play generator. Emits one text record per quiet position:
 * `fen ; stm_relative_score_cp ; wdl` (wdl in {1.0, 0.5, 0.0} from the side-to-move's POV).
 *
 * CLI: `./build/zenith-datagen <games> <out.txt> [seed] [nodes] [opening_plies]`
 */
int run_datagen(Engine *engine, int argc, char **argv);

/**
 * @brief Convert a bulletformat .data file (e.g. the public PlentyChess dataset) to Zenith's fen;score;wdl text.
 *
 * Each 32-byte record is stored from the side-to-move's perspective, so the reconstructed FEN is emitted
 * as "white to move" — that yields feature indices identical to Zenith's own convention.
 *
 * CLI: `./build/zenith-bullet2text <in.data> <out.txt> [max_records] [stride]`
 */
int run_bullet2text(int argc, char **argv);
