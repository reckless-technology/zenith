---
name: training-setup
description: Zenith's side of NNUE training — datagen, where the trainer now lives, GitHub remote and CI
metadata:
  type: reference
---

- **The trainer is not in this repository** (since 2026-10-31): it is `nnue-training`, the single source
  for the Reckless NNUE, and the PyTorch environment is made in *that* clone from its own
  `pytorch/requirements.txt`. Zenith's `tools/setup-dev --training` is gone with it, and every Python tool
  left in `tools/` is stdlib-only. The venv that used to live here was torch 2.13.0+cu130 with numpy 2.5,
  CUDA working on the RTX 4070.
- **Datagen:** `./build/zenith-datagen <games> <out.txt> [seed] [nodes] [openingPlies]` (built by `make datagen`) emits text
  `fen;score;wdl` (score is stm-relative cp, wdl is stm-relative in {0,0.5,1}). ~700 pos/s per core; fan
  out one process per core (32) with distinct seeds. Data goes in `data/` (gitignored).
- **Nets:** put a trained net in `nets/` (gitignored). Commit a chosen release net with `git add -f`.
- **GitHub:** remote `git@github.com:reckless-technology/zenith.git`. The `gh` CLI is available for setting up
  and inspecting CI (build + perft/bench tests) when the pipeline is ready.

Related: [[zenith-goal-independence]] [[pawnstar-reference-opponent]]
