---
name: zenith-goal-independence
description: Zenith's independence constraint (own arch, trainer, data, net format) is SUPERSEDED for the network and the net format - one shared NNUE evaluator across all the engines - and still stands for the trainer, the data and the search
metadata:
  type: project
---

Primary goal: make Zenith stronger than pawnstar C++ (the reference opponent). Pawnstar is a mature NNUE
engine (Lazy SMP, lockless TT, perspective net), so Zenith needs its own NNUE to compete — classical HCE
alone will not close the gap. **That goal was met:** Zenith beats pawnstar at every tested configuration
(~+135 Elo single-thread at 8+0.08, bookless both sides — see [[pawnstar-gap-benchmarks]]).

## The original constraint (2026-07-18)

**Independence was a hard constraint (user decision):** Zenith trains with its OWN PyTorch trainer (NOT
pawnstar's Rust `bullet`), on its OWN self-play data, with its own architecture (768→512 perspective) and
its own `.nnue` file format. Do not reuse pawnstar code, nets, or training data.

## Superseded for the network and the net format (2026-10-27)

Jonny: *"I would like us to use a common NNUE evaluator across the 3 pawnstar variants, zenith, and
stoopid chess. This will rationalize and simplify the process of developing new evaluation functions."*

So the network **shape** and the **net container** are now shared across every Reckless chess engine, and
an architecture change in one is a change owed to all of them. The specification is the `reckless-nnue`
skill; the engine side here is `src/nnue.c` plus `src/crc64.h`, the Python side `trainer/rnnue1.py`.

**Why the constraint could go, rather than merely being overridden.** Independence was never about
independence for its own sake — it was about being able to say Zenith beat pawnstar *fairly*, on its own
work rather than by borrowing pawnstar's. Two things retired that:

1. **The comparison is settled.** Zenith won it, measured, with its own architecture, trainer and data.
   Sharing an evaluator afterwards cannot retroactively make that result borrowed.
2. **Both engines now need retraining off the same withdrawn dataset.** Zenith's shipped `zenith-cap2`
   and pawnstar's shipped v12 both derive from the third-party dataset that was withdrawn, so both need a
   new net from a clean pool. Developing that evaluation function once instead of five times is the whole
   point, and the two engines were going to be retrained on a common pool regardless.

**And the shape chosen was Zenith's**, which is the clearest statement of what the independent work was
worth: its two additions over pawnstar's shape were measured and SPRT-gated here — material output
buckets **+14.7 Elo** and horizontal king mirroring **+14.3 Elo** — and the three pawnstar ports are
adopting them. The container chosen was pawnstar's, because `ZNNUE5` carried an 8-byte magic and told a
wrong net apart only by its file size, which is the exact failure the shared container exists to prevent.

## What still stands

- **The trainer is still Zenith's own** (`trainer/`, PyTorch/CUDA) — not pawnstar's `bullet`. Only the
  shape and the file format are common; nothing says the trainers must be.
- **The data is still Zenith's own** (`datagen`, or a public dataset converted by `zenith-bullet2text`).
- **No pawnstar source code is reused anywhere**, in the evaluator or outside it. Zenith remains a
  from-scratch C17 engine ([[pawnstar-reference-opponent]] is an opponent, not a dependency).
- **Never train on the withdrawn third-party dataset**, and do not extend `zenith-cap2`'s lineage. What
  becomes of that shipped net is Jonny's call, not an agent's.

**How to apply:** pipeline is `datagen/` → PyTorch trainer (`trainer/train.py`) → stamp into `RNNUE1`
(`tools/stamp_net.py`) → engine NNUE (`src/nnue.*`) → SPRT. Every strength change is still
perft/bench/SPRT-gated, and a change to the shape must land in the other engines too and reproduce their
bench signatures. Related: [[pawnstar-reference-opponent]] [[training-setup]] [[zenith-eval-experiments]]
