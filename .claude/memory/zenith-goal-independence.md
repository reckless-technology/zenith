---
name: zenith-goal-independence
description: Zenith's goal is to beat pawnstar C++ while staying fully independent from it (own trainer, data, arch, net format)
metadata:
  type: project
---

Primary goal: make Zenith stronger than pawnstar C++ (the reference opponent). Pawnstar is a mature NNUE
engine (Lazy SMP, lockless TT, v12 perspective net), so Zenith needs its own NNUE to compete — classical
HCE alone will not close the gap.

**Independence is a hard constraint (user decision, 2026-07-18):** Zenith trains with its OWN PyTorch
trainer (NOT pawnstar's Rust `bullet`), on its OWN self-play data (NOT pawnstar's ~5.7B-position dataset),
with its own architecture (768→512 perspective) and its own `.nnue` file format. Do not reuse pawnstar
code, nets, or training data.

**How to apply:** Pipeline is datagen ([src/datagen.cpp](../../src/datagen.cpp)) → PyTorch trainer
(trainer/train.py) → engine NNUE (src/nnue.*) → SPRT vs HCE, then vs pawnstar. Every strength change is
still perft/bench/SPRT-gated. Related: [[pawnstar-reference-opponent]] [[training-setup]]
