---
name: search-speed-levers
description: What actually moves Zenith's nps/strength — copy-make cost is the dominant per-node lever
metadata:
  type: project
---

Zenith uses **copy-make** (Position is a value type; search does `Position child = pos; child.make_move(m)`),
and Position embeds a **~2KB NNUE accumulator** (`values[2][512] int16` + king buckets). So every make_move copies
~2.2KB — this copy is the dominant per-node cost, and avoiding it is the biggest speed lever.

Wins realised (both SPRT-gated):
- **Copy-free legality oracle + prune-before-make: +66 Elo** (2026-07-21, commit 4183763). The move loop used to
  copy-make EVERY pseudo-legal move to test legality, THEN prune it with LMP/futility/SEE — paying the 2KB copy
  for moves never searched. Added `Position::is_legal_fast(m, checkers, pinned)` (pin/checker-aware, no copy;
  EP+castling defer to the exact copy-make path) + `LineBB`/`pinned_to_king()`. Compute checkers+pinned once per
  node, test legality BEFORE pruning, make_move only for survivors. Search tree is byte-identical (bench node
  signature unchanged) → pure speed: +22% nps at bench, MUCH more in real games (pruning-heavy trees). Validated
  by `./zenith legalcheck` (differential is_legal_fast==is_legal over a perft walk, 0 mismatch) + perft + ASan.
- Earlier: removing is_legal's redundant per-move copy-make (pseudo-legal + filter-in-search) was +126 Elo — same
  root cause (the 2KB accumulator copy).

**Next speed lever (biggest ceiling): make/unmake with an accumulator stack** — eliminates the per-node Position
copy entirely (DESIGN.md's aspirational "undo stack"). Large refactor touching every recursion site.

**How to apply:** when adding a per-move operation, ask "does this pay a make_move (2KB copy) it could avoid?"
Prune/filter before make_move wherever possible. Any change that alters the search tree must keep the bench node
signature meaningful (a pure-speed change keeps it IDENTICAL — a strong correctness check). See [[zenith-eval-experiments]].
