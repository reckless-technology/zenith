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
- **`~/pawnstar_nnue/data/*.data` holds an external bulletformat dataset that must NOT be used for
  training.** It is not pawnstar's own data, but it is off-limits all the same: do not convert, featurise
  or train on it, and do not reproduce a net from it. Zenith trains on its own self-play data.
- Off-limits for independence: pawnstar's engine code/nets and its `bullet` trainer (Zenith uses its own).

Related: [[zenith-goal-independence]] [[training-setup]]
