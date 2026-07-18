---
name: training-setup
description: Zenith NNUE training environment — venv, torch, datagen command, GitHub remote and CI
metadata:
  type: reference
---

- **PyTorch venv:** `/home/jonny/work/zenith/.venv` (torch 2.13.0+cu130, CUDA working on the RTX 4070,
  numpy 2.5). Activate with `. .venv/bin/activate`. Gitignored.
- **Datagen:** `./zenith datagen <games> <out.txt> [seed] [nodes] [openingPlies]` emits text
  `fen;score;wdl` (score is stm-relative cp, wdl is stm-relative in {0,0.5,1}). ~700 pos/s per core; fan
  out one process per core (32) with distinct seeds. Data goes in `data/` (gitignored).
- **Nets:** trainer writes to `nets/` (gitignored). Commit a chosen release net with `git add -f`.
- **GitHub:** remote `git@github.com:jonny-reckless/zenith.git`. The `gh` CLI is available for setting up
  and inspecting CI (build + perft/bench tests) when the pipeline is ready.

Related: [[zenith-goal-independence]] [[pawnstar-reference-opponent]]
