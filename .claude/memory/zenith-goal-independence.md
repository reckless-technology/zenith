---
name: zenith-goal-independence
description: Zenith's independence constraint (own arch, trainer, data, net format) is SUPERSEDED for the network, the net format AND the trainer - one shared NNUE evaluator and one shared trainer in nnue-training - and still stands for the search and the engine source
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
skill; the engine side here is `src/nnue.c` plus `src/crc64.h`, and the Python side is
`nnue-training/spec/rnnue1.py`, carried here verbatim as `tools/rnnue1.py` (SHA-256
`90472ea1863cd394faac23f8175761bbc9003bda36ccf75ba6f747e5d116e3fa`) because three tools in `tools/`
import it and must run on a CI runner with no network — change it in both repositories or in neither.

**Why the constraint could go, rather than merely being overridden.** Independence was never about
independence for its own sake — it was about being able to say Zenith beat pawnstar *fairly*, on its own
work rather than by borrowing pawnstar's. Two things retired that:

1. **The comparison is settled.** Zenith won it, measured, with its own architecture, trainer and data.
   Sharing an evaluator afterwards cannot retroactively make that result borrowed.
2. **Both engines now need retraining off the same dataset — the public PlentyChess dataset, by
   Yoshie2000, which is no longer used.** Zenith's shipped `zenith-cap2` and pawnstar's shipped v12 both
   derive from it, so both need a new net from a clean pool. Developing that evaluation function once instead of five times is the whole
   point, and the two engines were going to be retrained on a common pool regardless.

**And the shape chosen was Zenith's**, which is the clearest statement of what the independent work was
worth: its two additions over pawnstar's shape were measured and SPRT-gated here — material output
buckets **+14.7 Elo** and horizontal king mirroring **+14.3 Elo** — and the three pawnstar ports are
adopting them. The container chosen was pawnstar's, because `ZNNUE5` carried an 8-byte magic and told a
wrong net apart only by its file size, which is the exact failure the shared container exists to prevent.

## What still stands

- **The trainer is shared too, as of 2026-10-31, and `trainer/` is GONE from this repository.**
  `nnue-training` (public) was created as the single source for the NNUE — architecture, container, both
  trainers, data pipeline — and Zenith's `trainer/` was a *duplicate* of it: `rnnue1.py` byte-identical
  to `spec/rnnue1.py`, and `features.py`, `train.py` and `verify.py` one added `import spec_path` line
  away from `pytorch/`'s. Maintaining it in two places is what that repository exists to end, so it was
  removed and the documents now point there. The split kept is **producer asserts, consumer verifies**:
  the trainers and the container are theirs; the loader (`src/nnue.*`), the net tools in `tools/`,
  `nnuecheck` and the "does this net load *and play* here" gates are Zenith's. `tools/setup-dev`'s
  `--training` / `-Training` option went with it — make the training environment in a `nnue-training`
  clone, from its own `pytorch/requirements.txt`.
  From 3.0.0 the embedded net is `nets/reckless-lc0-runf-sb870.nnue`, trained **entirely on Leela Chess
  Zero self-play data** in the shared Reckless NNUE pipeline (pawnstar's `bullet`), adopted because it
  replaces the PlentyChess lineage, and the swap did **not** wait on an A/B result — removing the data
  dependency is the point. It loads unchanged — identical architecture and `RNNUE1`
  header, `nnuecheck` 3/3 at `max|diff| 0` and the shared `pytorch/verify.py` 0 cp over 4,000 positions — and moves
  the bench signature 1,390,606 -> 1,653,778. **Its strength in Zenith is NOT YET MEASURED** (Jonny is
  running `reckless-lc0-runf-sb870` against `zenith-cap2` himself); the +9.99 Elo at 8+0.08 and +1.13 at
  12+0.12 measured for this net are **pawnstar's numbers against pawnstar's old net** and do not transfer.
- Zenith's `datagen` self-play generator trained only the `v1`-`v5` pilots and the opening book. **No net
  Zenith ever released was trained on it** — the **PlentyChess** dataset was adopted at `pc1` because it
  beat the self-play plateau by ~154 Elo, and every released net from `pc2` to `cap2` descends from it. A
  claim that Zenith's nets have always come from its own self-play data is **false**. From 2026-10-25 to
  2026-10-30 a grep for the name found nothing, because commit `8900c39` scrubbed it — evidence of the
  name's absence, not the dependency's; that scrub is **reversed**, the name is back in the documents and
  in `datagen/`, and a grep is a working way to find this dependency again.
- **No pawnstar source code is reused anywhere**, in the evaluator or outside it. Zenith remains a
  from-scratch C17 engine ([[pawnstar-reference-opponent]] is an opponent, not a dependency).
- **Never train on the PlentyChess dataset**, and do not extend `zenith-cap2`'s lineage — the 3.0.0 net
  replaces it. **`nets/zenith-cap1.nnue` and `nets/zenith-cap2.nnue` are removed from the tree** (3.0.0;
  history preserves them), so no release ships a PlentyChess-derived net — do not restore either, and do
  not point a build default or a test fixture at one. The **attribution stays** in README.md and
  NNUE_TRAINING.md deliberately: those nets were measurably stronger for that data and the record should
  show where they came from. Credit and fact only — no rights or legal argument anywhere.

**How to apply:** pipeline is `datagen/` → the shared trainer in `nnue-training` → stamp into `RNNUE1`
(`tools/stamp_net.py`) → engine NNUE (`src/nnue.*`) → SPRT. Every strength change is still
perft/bench/SPRT-gated, and a change to the shape must land in the other engines too and reproduce their
bench signatures. Related: [[pawnstar-reference-opponent]] [[training-setup]] [[zenith-eval-experiments]]
