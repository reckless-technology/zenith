---
name: zenith-eval-experiments
description: Results log of NNUE eval-architecture experiments (output buckets, king buckets, ...)
metadata:
  type: project
---

Running log of Zenith NNUE eval experiments (all SPRT-gated vs the current best net). Goal: stronger eval to close the single-thread gap to pawnstar (see [[pawnstar-gap-benchmarks]]).

## Output buckets (8, by piece count) — NEUTRAL, not shipped (2026-07-20)
- 8 output heads selected by `clamp((pieceCount-2)/4, 0, 7)`. Full impl (ZNNUE2 magic), 0cp gate + nnuecheck PASS.
- Trained `zenith-ob1` (512, 14ep, wdl 0.3) on the SAME 187M plentychess cache/recipe as pc2.
- **Val loss IMPROVED: 0.015925 vs pc2's 0.016161 (~1.5% better MSE fit).**
- **BUT SPRT vs pc2 (tc=8+0.08, 3000 games): −5.9 ± 8.4 Elo (49.15%), LLR never neared H1.** Neutral-to-slightly-negative.
- **Lesson: lower val loss did NOT translate to Elo.** Output buckets add eval discontinuities at piece-count
  boundaries (a capture crossing a threshold jumps the eval), which can add search noise and cancel the fit gain.
  At 512 hidden / 190M positions, output buckets don't pay off in isolation. Shelved, not deleted.
- Work preserved: `scratchpad/output_buckets.patch` + `scratchpad/zenith-ob1.nnue`. Output buckets typically help
  more layered on king-bucketed inputs / bigger nets — revisit ON TOP of king buckets if those win.

## King buckets (8, file-pairs × board-halves) — IN PROGRESS
- Design: `scratchpad/king_buckets_design.md`. FT grows [768]→[8*768]=[6144][512]; refresh-on-king-move when the
  moving king crosses a bucket boundary. Bigger expected lever (input-side enrichment). ZNNUE3 magic.

**How to apply:** Never trust val loss alone — Elo via SPRT is the only verdict (CLAUDE.md discipline). Isolate one
architecture change per SPRT for clean attribution.
