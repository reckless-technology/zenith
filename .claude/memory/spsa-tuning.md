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
