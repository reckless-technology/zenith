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
  All gates PASS (perft, 0cp all 8 buckets, nnuecheck incl. castling, color-mirror symmetry). Design:
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

## Pawnstar-idea adoption experiments (2026-07-23/24) — 1 shipped, 2 neutral
After the corrected benchmarks showed Zenith AHEAD of pawnstar, we tried adopting the three techniques pawnstar
has that Zenith lacked:
1. **Checking-aware LMP/futility** (never prune checking quiets): SPRT **NEUTRAL** (−3.0 ± 14.1 @ 822 games,
   cancelled). Zenith's SPSA-tuned margins already price in check-blind pruning; pawnstar needs the exemption
   because its cruder untuned pruning over-prunes. KEPT the infrastructure: `gives_check_fast` +
   `discovered_check_candidates` (differentially validated in legalcheck, 0 mismatches / 7.26M nodes, in CI) —
   reusable for future check-aware ideas (e.g. quiet checks in qsearch).
2. **8MB eval cache** (2^20 × u64, key-high-bits verify ⇒ false hits impossible, Lazy-SMP-safe): **SHIPPED**
   (2d7e581). Behaviour-identical, +0.1% (midgame) to +2.2% (endgame) nps — main win is qsearch stand-pat.
3. **TT-move-before-movegen** (hash-trusted TT move searched before generating; structural 4-compare guard
   instead of is_pseudo_legal): **REVERTED**. Two findings: (a) the speed win is ~ZERO — magic-bitboard movegen
   isn't a real cost at TT-cutoff nodes; (b) it is NOT behaviour-identical — move scoring then runs AFTER the
   TT subtree, so siblings get ordered with fresher killers/history (inherent tension: can't skip movegen AND
   score with pre-search state; killers could be snapshotted, the history tables can't). SPRT of the combined
   change trended neutral-negative (−4.7 @ ~300 games, cancelled).

**Meta-lesson: an engine's techniques don't transfer just because the engine is strong.** Pawnstar's tricks
compensate for ITS weaknesses (crude LMR, no futility/SEE-prune/aspiration/corrhist). Zenith's tuned search
already extracts what those tricks provide. Adopt ideas that fix a measured cost (eval cache → uncached qsearch
stand-pat), not ideas that exist in the other engine.

**How to apply:** Never trust val loss alone — Elo via SPRT is the only verdict (CLAUDE.md discipline). Isolate
one architecture change per SPRT. Diagnose neutral timed results with a FIXED-DEPTH match to split eval-quality
from speed before shelving; if the eval is genuinely better, scale data before abandoning — but watch for
diminishing data returns and pivot to speed once eval scaling flattens. See [[pawnstar-gap-benchmarks]].

**Improving heuristic: NO GAIN (2026-07-26).** Standard is-eval-better-than-2-plies-ago gating RFP margin
(depth-1 when improving) + LMP quota (halved when not): bench nodes -33% at depth 13, but SPRT [0,5] @
8+0.08 vs pre-change baseline capped at 4000 games WITHOUT verdict: +1.74 ± 6.32 Elo, LLR 0.08. Early +8
readings regressed to ~+2 as the sample grew. Not landed (discipline: unproven = no). Zenith's tuned static
margins apparently already capture most of what improving-conditioning buys elsewhere; a retry should
co-tune the margins with SPSA rather than bolting improving onto margins tuned without it.

**qsearch TT: +17.0 ± 7.6 Elo, H1 accepted (2026-07-26).** Probe at qsearch entry (non-PV bound cutoffs —
any hit suffices at depth 0), stand-pat seeded from entry eval, store at depth 0 with fail-soft bounds.
Found via the observation that qsearch prefetched the TT but never read it. Bench signature 2657379 →
2264816 (-15% nodes). Landed as PR #3. Contrast with the improving heuristic (same day, unproven): the
biggest wins remain plugging MISSING standard machinery, not re-conditioning what SPSA already tuned.

**Output buckets (ob2): +14.7 ± 7.7 Elo fixed-depth SPRT over kb3, H1 accepted (2026-07-27).** Phase 1 of
the evaluator project: 8 material heads ((pieces-2)/4) on the unchanged kb 768×8→512 transformer, retrained
on the same 16 shards (8 epochs, val 0.01483 vs kb3-era ~0.0156). ZNNUE4 format; v3 nets broadcast at load.
Training: ~32 min/epoch on the 4070 (~723k pos/s). Next: horizontal king mirroring (phase 2).

**King mirroring infrastructure (ZNNUE5) landed (2026-07-27, PR #13).** Phase 2: perspective with own king
on files e-h flips horizontally (square^7) onto a-d; 8 king buckets = 4 files × 2 halves of the folded
region; finny cache keyed (perspective, mirror, bucket); d/e king crossing = refresh even when the bucket
number is unchanged. Loader keeps v4/v3 back-compat; verify.py is current-format-only. All gates 0-diff;
bench unchanged; node-identical to main with ob2. **Implementation trap that cost a debugging round:** the
mirror flip must be applied in ALL THREE accumulator primitives (add/remove/move feature) — the first
attempt flipped only move_feature, and nnuecheck caught millions of lane mismatches from startpos (both
kings on e-file ⇒ both perspectives mirrored). The shard .npz caches bake in feature indices, so a feature
contract change requires full refeaturisation (~2.5h, 16 shards; s08-s15 texts regenerate from
~/pawnstar_nnue/data binpacks 13148/13227/13247/13349/13364/13381/13399/13419 with
`bullet2text IN OUT 0 4` — maxRecords=0, **stride 4**; verified old row counts == ceil(records/4) — see
scratchpad refeat_v5.sh). Long-running background jobs must be launched with `setsid nohup` so a Claude
Code harness restart cannot kill them (the first refeaturisation run died this way mid-shard). Net
zenith-km1 training on the same ob2 recipe; SPRT vs ob2 pending.

**Embedded net + HCE removal (2026-07-28).** The shipped net is embedded into every binary at build time
(tools/embed_net.py -> generated C array -> build/embedded_net.o, cached; +3.8s once per net change, zero
on normal rebuilds). engine_new() loads it by default; EvalFile swaps nets at runtime and falls back to the
embedded copy on load failure; empty EvalFile returns to it. The PeSTO HCE was removed (git history keeps
it) — eval is NNUE always. **The bench signature moved 2,264,816 -> 1,469,216 by necessity:** the old
signature was HCE search trees (bench ran netless); with the net as default, bench trees are NNUE-scored.
Search itself proven untouched: node-identical to pre-change main with the same EvalFile, and embedded ==
file-loaded ob2. The signature now guards loader + embedded weights too. Same number on clang/gcc/ASan/PEXT.
Datagen self-play now emits NNUE-scored records by default (matters for phase 4).

**King mirroring net (km1): +14.3 Elo fixed-depth SPRT over ob2, H1 accepted at 3,968 games (2026-07-28).**
Phase 2 payoff: the ZNNUE5 mirrored architecture (PR #13) trained on the SAME 1.4B PlentyChess positions
refeaturised under the mirrored contract, exact ob2 recipe (8 epochs, batch 32768, lr 1.2e-3, wdl 0.3).
Val 0.014691 — project best (ob2 0.01483). LLR 2.96, 52.05%. Watch-out that resolved itself: a val-loss
bump at epoch 5 (0.0155->0.0162) from streaming shard order recovered fully under lr decay by epoch 7 —
don't panic-stop on a single-epoch val regression. Shipped as the embedded default (Makefile NET; bench
re-pinned 1,469,216 -> 1,421,091). Architecture lever total so far: ob2 +14.7, km1 +14.3 on top —
input-side AND output-side enrichment both paid off once data was at 1.4B scale. Next: capacity (512->1024).
