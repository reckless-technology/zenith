---
name: zenith-nnue-pilot-status
description: NNUE net progression (pilot -> v5) and current strength vs HCE and pawnstar
metadata:
  type: project
---

The NNUE pipeline (datagen -> PyTorch train -> quantised `.nnue` -> engine AVX2 full-refresh eval) is
CORRECT and validated: 0cp gate (engine == trainer, bit-identical), colour-mirror symmetric, finds tactics.

**Net progression (fixed-depth-8 Elo vs the PeSTO HCE, unless noted):**
- pilot (4.1M random-ply, HCE labels): **−325** — underfit + noisy (0.92 corr, ~267cp noise; minimax amplifies).
- v2 (19.5M): −38. v4 (46M = v2+v3 book): **+139** (first to beat HCE); −28 TIMED. 768 hidden OVERFITS (512 is right).
- **v5 (72M, incl. v4-NNUE-labelled self-play data): +301 fixed / +166 TIMED vs HCE; +211 vs v4; −377 vs pawnstar.**

**Key lessons:** (1) more data is the dominant lever; (2) self-play (net-in-the-loop datagen) COMPOUNDS
hugely — one iteration added +211 Elo and flipped the timed result positive; (3) 512 hidden is the sweet
spot for this data (768 overfits); (4) speed matters — int16+AVX2 refresh (641k->1.08M nps) added ~+41 timed.

**Current best net: `nets/zenith-v6.nnue` (committed).** Speed: AVX2 refresh + incremental accumulator
(embedded in Position, `nnuecheck` verifies incremental==refresh) + pseudo-legal movegen with legality
filtered in-search (removes is_legal's redundant copy-make — SPRT +126 Elo, synergistic with the
accumulator since is_legal was copying+updating the 2KB accumulator per move). nps 641k→1.08M→1.98M.

**Gap to pawnstar (timed 8+0.08): −489 (v4) → −385 (v6 self-play) → −308 (v6 after speedup).** v6 beats HCE
+223 timed. Next levers: singular extensions, more self-play iterations, Lazy SMP + lockless TT, SPSA tuning.
Related: [[zenith-goal-independence]] [[training-setup]]
