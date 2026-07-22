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

## King buckets (8, file-pairs × board-halves, ZNNUE3) — SPEED-BOUND, +4 Elo (2026-07-20)
- FT [768]→[6144][512] (8× rows); refresh-on-king-move when the moving king crosses a bucket boundary.
  All gates PASS (perft, 0cp all 8 buckets, nnuecheck incl. castling, colour-mirror symmetry). Design:
  `scratchpad/king_buckets_design.md`. Trained `zenith-kb1` (pc2 recipe): **best val loss yet, 0.01575**.
- **Timed SPRT vs pc2: −0.2 ± 8.5 (NEUTRAL).** BUT **fixed-depth 9: +21.3 ± 18.8 Elo** → the eval is
  genuinely stronger; the 8× FT (6.3MB, blows cache) + king-move refresh cost the speed back. nps hit:
  ~3% opening, ~6% midgame, **~20% endgame** (active kings ⇒ frequent bucket-crossing refreshes).
- **Finny refresh cache** (thread_local per (perspective,bucket): cached accumulator + board it was built
  from; refresh applies only piece-diffs, net-generation-guarded). Byte-identical (all gates still PASS).
  Cut avg slowdown to ~6% (suite depth-18: 1.65M vs pc2 1.75M nps). **Re-SPRT vs pc2: +3.9 ± 8.6, LLR
  +0.40** — positive, best net measured, but NOT a decisive pass (effect ~+4 sits below the elo1=5 bound).
- **Lesson:** input-side eval enrichment (king buckets) DOES add real eval strength (+21 fixed-depth),
  unlike output buckets — but a bigger net's speed cost caps the timed payoff. Residual ~6% is FT
  cache-locality on every incremental update (unavoidable at 8× FT size). Engine+trainer code is correct &
  ready (uncommitted on main). To convert the +21 eval headroom into a decisive timed win, need more eval
  strength that outpaces the speed cost → **more training data** (no speed cost). Blocked by RAM: the
  all-in-RAM featuriser caps at ~230M positions on this 62GB box; going bigger needs a streaming/mmap trainer.

## King buckets + BIG DATA (zenith-kb2) — SHIPPED, +57 Elo (2026-07-21)
- Root cause of kb1's marginal gain was NET STRENGTH, not the architecture: king buckets give eval headroom
  (+21 fixed-depth) but at 190M positions the eval edge barely beat the ~6% speed cost. Fix = more data
  (no speed cost). Refactored the trainer for scale: chunked `--featurise-shard` (pre-allocate + fill, ~13GB
  RAM/proc, parallelizable) + `--shard-dir` streaming training (one ~95M-position shard in RAM at a time,
  shard-order + in-shard shuffle). Lets us train past the ~230M all-in-RAM ceiling on this 62GB box.
- Trained `zenith-kb2`: 8 PlentyChess shards = **745M positions** (7 train ~650M + 1 val held out), 8 epochs,
  same recipe (512, wdl 0.3, lr 1.2e-3). **Best val loss yet: 0.015417** (pc2 0.01616, kb1 0.01575).
- **SPRT vs pc2: +57 Elo, H1 ACCEPTED at 954 games (58.1%, LLR 2.96).** Decisive — the biggest single win
  of the project. king buckets (2M+ RAM finny cache) + 3.4× data. Shipped as the default net; ZNNUE1
  nets (pc1/pc2) retired (engine is ZNNUE3-only).
- **Lesson: when an architecture wins at fixed depth but not timed, the lever is usually MORE DATA (free
  eval strength, zero speed cost), not more architecture.** Data scale >> capacity tweaks at this stage.

## More data: kb3 (1.4B positions) — +8 Elo, DIMINISHING RETURNS (2026-07-22)
- Trained `zenith-kb3` on all 16 PlentyChess shards (~1.4B positions, 2x kb2's 650M), same recipe, 6 epochs.
  Val loss 0.015195 (kb2 0.015417, pc2 0.016161). **SPRT vs kb2: +7.9 ± 7.4 (CI just excludes 0), LLR +1.33.**
- **Data returns are diminishing:** 190M→650M gave +57 (kb2), 650M→1.4B gave only +8 (kb3). The PlentyChess
  data lever is largely tapped out. Shipped kb3 anyway (strictly-better net, same size/speed = free upgrade),
  but further gains need a DIFFERENT lever: speed (make/unmake, transfers ~1:1 to pawnstar) or better/own data.
- The streaming trainer (`--shard-dir`) + parallel `--featurise-shard` handled 1.4B on a 62GB box fine.

**How to apply:** Never trust val loss alone — Elo via SPRT is the only verdict (CLAUDE.md discipline). Isolate
one architecture change per SPRT. Diagnose neutral timed results with a FIXED-DEPTH match to split eval-quality
from speed before shelving; if the eval is genuinely better, scale data before abandoning — but watch for
diminishing data returns and pivot to speed once eval scaling flattens. See [[pawnstar-gap-benchmarks]].
