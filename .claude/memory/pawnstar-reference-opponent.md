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
- **`~/pawnstar_nnue/data/*.data` is the PUBLIC PlentyChess dataset** (bulletformat, 5.7B positions, from
  huggingface Yoshie2000/plentychess_data_bulletformat) — NOT pawnstar's private data. It is fine for
  Zenith to train on (public third-party data, independent of the pawnstar engine); the user directed it.
  Use `./build/zenith-bullet2text <in.data> <out.txt> [max] [stride]` (built by `make datagen`) to convert to Zenith's fen;score;wdl.
- Off-limits for independence: pawnstar's engine code/nets and its `bullet` trainer (Zenith uses its own).

Related: [[zenith-goal-independence]] [[training-setup]]
