---
name: spsa-tuning
description: SPSA search-param tuning harness (tools/spsa.py) — how to run it; first pass gave +26.5 Elo
metadata:
  type: reference
---

Zenith has an SPSA tuner for its hand-set search constants (fastchess has no built-in SPSA).

- **Engine side:** 11 search params are exposed as UCI spin options via `SearchParams` (search.h) + `set_search_param`
  (search.c). Option NAMES are PascalCase strings (`RfpMargin`, `LmrDivisor`, `HistoryMax`, …) even though the engine
  members are snake_case — spsa.py and GUIs depend on those strings, so don't rename them. Defaults reproduce the
  shipped engine; changing a param changes the search (verify via a fixed-depth node count).
- **Tuner:** `tools/spsa.py` — Spall SPSA with decaying gains, perturbs all params +/- c_k*Delta, plays a theta+ vs
  theta- self-play match per iteration (same binary, different `option.X=` values), parses fastchess
  `Games/Wins/Losses`, checkpoints theta to json each iteration. Run:
  `python tools/spsa.py --engine ./zenith --net nets/zenith-kb3.nnue --openings ~/pawnstar_nnue/openings.epd \
   --fastchess ~/pawnstar_nnue/fastchess/fastchess --iters 800 --games 16 --concurrency 16 --tc 5+0.05 --out state.json`
- **Timing:** 16 games/iter at conc 16, tc 5+0.05 ≈ 27s/iter → 800 iters ≈ 6h. tc 8+0.08 is ~2x slower (impractical
  for many iters) — tune at a FAST tc (5+0.05), it transfers to the 8+0.08 validation.
- **Validation is mandatory:** SPSA optimizes a noisy fast-tc objective. Always SPRT the tuned OPTION values vs the
  defaults at 8+0.08 (same binary, different options — clean isolation) before baking. Then bake winners into
  `SearchParams` defaults and consistency-check (baked node signature == tuned-options node signature).

**First pass (2026-07-23, commit a8113c0): +26.5 ± 9.6 Elo, H1 accepted.** Biggest moves: RfpMargin 80→62, LmrBase
80→86, FutilityMargin 90→96, HistoryMax 400→418, LmpBase 3→4; the other 6 were already near-optimal. Re-tuning
periodically (esp. after eval/net changes) is worth it; the harness is reusable. See [[zenith-eval-experiments]].

**Second pass (2026-07-26): +11.1 ± 6.0 Elo, H1 accepted (LLR 2.96, ~2900 games @ 8+0.08).** Run as ONE
chained descent via the new `--resume` flag (continues theta AND iteration count so the gain schedule keeps
decaying): 2 rounds x 250 iters, 16 games/iter @ 5+0.05, conc 16 (~1.9h/round). Theta converged by ~350
iters (last 150 = noise). Winners: RfpMargin 62→56, NmpDivisor 202→199, LmpBase 4→5, FutilityBase 102→94,
FutilityMargin 96→98, SeeCaptureMargin 102→97, LmrBase 86→90, LmrDivisor 229→220, HistoryMax 418→402;
SingularMargin/AspirationDelta never moved. New bench signature 2657379. Consistency gate (baked ==
options, node-identical) passed. Ops lesson: launch the tuner detached (setsid) in its OWN Bash call —
never put pkill and the launch in one command (the pattern matches the launch line's text and kills the
wrapper); monitor liveness by checkpoint-file mtime, never pgrep.

**Pass 3 (2026-07-31, cap1-era): NO BAKE — optimum confirmed stable.** 800 iterations at 8+0.08 against
the 1024-hidden engine; converged theta moved only a few percent of range (NmpDivisor 199->186,
FutilityMargin 98->107, LmrBase 90->84, LmrDivisor 220->230, SeeCaptureMargin 97->102, AspirationDelta
21->19; four params returned exactly to defaults). Validation SPRT [0,5] ran the full 10,000-game cap:
50.07%, +0.5 +/- 4.8 Elo, LLR -0.70 — unproven, defaults stand. Conclusion: the pass-2 parameters are
robust across BOTH the mirroring and the 2x-capacity eval changes; SPSA is tapped out for this engine
generation (pass 1 +26.5, pass 2 +11.1, pass 3 ~0). Next tuning should wait for a structural search
change, not an eval swap.
