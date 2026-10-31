---
name: pawnstar-reference-opponent
description: Locations of the pawnstar reference opponent, SPRT harness, GPU, and training tooling on this machine
metadata:
  type: reference
---

Reference opponent + tooling on this box:
- **Pawnstar C++** engine: `/home/jonny/work/pawnstar` (build binary `build/pawnstar`, embedded net
  `nnue/pawnstar-v12.bin`; header-only C++23, Lazy SMP, lockless TT). Siblings: `pawnstar-go`, `pawnstar-rs`.
- **SPRT harness:** `~/pawnstar_nnue/fastchess` (fastchess binary). Openings book
  `~/pawnstar_nnue/openings.epd` (174MB EPD). NOTE: `cutechess-cli` is NOT installed, but `tools/sprt.sh`
  assumes it — use fastchess or adapt the harness.
- **Hardware:** NVIDIA RTX 4070 Laptop GPU (8GB); 32 CPU cores; clang++ 18, g++ 13.
- **`~/pawnstar_nnue/data/*.data` holds the public PlentyChess bulletformat dataset (by Yoshie2000), which
  must NOT be used for training.** It is not pawnstar's own data, and it is off-limits: do not convert,
  featurise or train on it, and do not reproduce a net from it. Zenith's nets up to `zenith-cap2` were
  trained on it; the net shipped from 3.0.0 is trained on Leela Chess Zero self-play data instead.
- Off-limits: pawnstar's engine code and its nets. **Its `bullet` trainer is no longer off-limits** — the
  trainers were gathered into `nnue-training` as the shared single source (and Zenith's duplicate
  `trainer/` was removed on 2026-10-31), so bullet is the production route for every Reckless net now.
  The engine source stays independent; see [[zenith-goal-independence]].

Related: [[zenith-goal-independence]] [[training-setup]]
