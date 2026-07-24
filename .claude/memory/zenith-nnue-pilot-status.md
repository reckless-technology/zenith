---
name: zenith-nnue-pilot-status
description: NNUE net progression (pilot -> v5) and current strength vs HCE and pawnstar
metadata:
  type: project
---

The NNUE pipeline (datagen -> PyTorch train -> quantised `.nnue` -> engine AVX2 full-refresh eval) is
CORRECT and validated: 0cp gate (engine == trainer, bit-identical), color-mirror symmetric, finds tactics.

**Net progression (fixed-depth-8 Elo vs the PeSTO HCE, unless noted):**
- pilot (4.1M random-ply, HCE labels): **−325** — underfit + noisy (0.92 corr, ~267cp noise; minimax amplifies).
- v2 (19.5M): −38. v4 (46M = v2+v3 book): **+139** (first to beat HCE); −28 TIMED. 768 hidden OVERFITS (512 is right).
- **v5 (72M, incl. v4-NNUE-labelled self-play data): +301 fixed / +166 TIMED vs HCE; +211 vs v4; −377 vs pawnstar.**

**Key lessons:** (1) more data is the dominant lever; (2) self-play (net-in-the-loop datagen) COMPOUNDS
hugely — one iteration added +211 Elo and flipped the timed result positive; (3) 512 hidden is the sweet
spot for this data (768 overfits); (4) speed matters — int16+AVX2 refresh (641k->1.08M nps) added ~+41 timed.

**Current best net: `nets/zenith-pc2.nnue` (committed, 512 hidden).** Trained on the public PlentyChess
dataset (via `bullet2text`), which broke the self-play plateau: pc1 (70M) was +154 over the best self-play
net v6; pc2 (187M) +30 more. More data then saturated the 512 net.

**512 is the sweet spot (do not chase bigger nets here):** a 1024 net (pc3) on the same 187M was TIED at
fixed depth and LOST timed (−54 vs pc2 vs pawnstar), because 1024 costs −37% nps (1.98M→1.25M) and speed
dominates. Trainer uses EmbeddingBag (memory-efficient) + int16 RAM to reach 200M+ positions on 8GB.

**Speed stack:** AVX2 refresh + incremental accumulator (Position-embedded, `nnuecheck` verifies
incremental==refresh) + pseudo-legal movegen with legality filtered in-search (SPRT +126 Elo; removes
is_legal's redundant copy-make). nps 641k→1.98M. Search: + IIR + singular extensions.

**Gap to pawnstar (timed 8+0.08): −489 (v4) → −385 → −308 (speed) → −246 (pc1) → −211 (pc2).** Halved.
Next levers: more PlentyChess shards (200M→500M+ via streaming), SPSA tuning of search margins for the
NNUE eval scale, Lazy SMP + lockless TT.
Related: [[zenith-goal-independence]] [[training-setup]]
