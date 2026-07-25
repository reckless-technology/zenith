---
name: setwise-elegance-nonregression-gate
description: "Jonny values architectural elegance (e.g. setwise bitboard movegen) and ships such refactors on a NON-regression SPRT, not a gain test"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 221925ef-3c7d-4275-96ac-0cd7362cb511
  modified: 2026-07-25T05:55:36.169Z
---

When a refactor is architecturally preferable but strength-neutral in intent (example: the setwise
bitboard-shift pawn move generation replacing the per-pawn loop, July 2026), Jonny wants it shipped if it
is *not a regression* — the SPRT gate is `ELO0=-5 ELO1=0` (accepting H1 ⇒ keep), not the default gain
bounds `[0, 5]`.

**Why:** the default SPRT bounds test "is this a gain?", which rejects perfectly neutral changes; Jonny
explicitly said "architecturally i prefer the elegance of the setwise implementation" while a neutral
gain-test was mid-run, so the correct hypothesis is non-regression.

**How to apply:** for elegance/simplification refactors that perturb the bench signature only via move
order or tie-breaking (no intended strength change), run `ELO0=-5 ELO1=0 tools/sprt.sh`; reserve the
default `[0, 5]` bounds for changes meant to gain Elo. See [[spsa-tuning]] and [[zenith-eval-experiments]]
for the gain-test side of the discipline.
