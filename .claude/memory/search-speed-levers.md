---
name: search-speed-levers
description: What actually moves Zenith's nps/strength — copy-make cost is the dominant per-node lever
metadata:
  type: project
---

Zenith uses **copy-make** (Position is a value type; search does `Position child = pos; child.make_move(m)`),
and Position embeds a **~2KB NNUE accumulator** (`values[2][512] int16` + king buckets).

**CORRECTED per-node cost model (2026-07-22): the 2KB accumulator MEMCPY is cheap; the real cost is the
incremental accumulator UPDATE — reading feature-transformer columns from the 6.3MB king-bucketed FT (cache
misses).** So the biggest per-node lever is avoiding the *whole* make_move (copy + FT reads), not just the copy.
- **make/unmake was tried and REVERTED (negative result):** replacing copy-make with in-place make/unmake +
  reverse-delta unmake was byte-identical (bench 4068365, movecheck/perft/nnuecheck all pass) but **7-12% SLOWER
  with the net loaded** — unmake re-reads the FT columns (reverse deltas), doubling the expensive part to save a
  cheap memcpy. Accumulator-stack (option b) wouldn't help either: it still pays copy(2KB)+forward FT reads.
  Design/notes: `scratchpad/make_unmake_design.md`. Don't retry unless the FT shrinks / cache behaviour changes.

Wins realised (both SPRT-gated):
- **Copy-free legality oracle + prune-before-make: +66 Elo** (2026-07-21, commit 4183763). The move loop used to
  copy-make EVERY pseudo-legal move to test legality, THEN prune it with LMP/futility/SEE — paying the FULL
  make_move (copy + FT-read incremental update) for moves never searched. Added `Position::is_legal_fast(m,
  checkers, pinned)` (pin/checker-aware, no copy;
  EP+castling defer to the exact copy-make path) + `LineBB`/`pinned_to_king()`. Compute checkers+pinned once per
  node, test legality BEFORE pruning, make_move only for survivors. Search tree is byte-identical (bench node
  signature unchanged) → pure speed: +22% nps at bench, MUCH more in real games (pruning-heavy trees). Validated
  by `./zenith legalcheck` (differential is_legal_fast==is_legal over a perft walk, 0 mismatch) + perft + ASan.
- Earlier: removing is_legal's redundant per-move copy-make (pseudo-legal + filter-in-search) was +126 Elo — same
  root cause (the 2KB accumulator copy).

More wins realised (2026-07-24, from the independent-review speed pass):
- **AVX2-vectorized NNUE forward pass: +8.5 ± 5.8 Elo** (commit 851bb83). `nnue_evaluate`'s SCReLU dot was a
  scalar 512-wide loop run at every `evaluate()` (qsearch stand-pat + leaves). Vectorized: 16-wide int16 clamp
  → int32 widen → exact int32 clamped*weight → int64 term accumulation via `mul_epi32` (reads low-32 of each
  64-bit lane as signed; even/odd int32 lanes via one 32-bit shift). BIT-EXACT (0cp gate passes; identical node
  count at fixed depth) → measured **+5.2%** at identical work (fixed-depth-18 time, pinned P-core). Lesson: the
  forward pass, though "only per-eval", is a real fraction of time with a net loaded; SIMD it.
- **Position struct shrink 2464→2240 bytes** (commit bd0c9d3): mailbox `board[64]` as uint8 (not the 4-byte
  Piece enum, −192) + packed scalars (−32). Behaviour-identical. NOTE: pure field *reordering* saves nothing —
  the accumulator's `_Alignas(32)` rounds the struct to a 32-byte multiple, so removed internal padding just
  becomes trailing padding; the mailbox WIDTH is the lever. The accumulator's 2048 bytes of int16 `values` are
  irreducible (they hold feature-column sums that overflow int8 — the accumulator can NOT be int8; only the
  hidden-layer matmul could use int8/VNNI, but SCReLU + a 1024→1 output make that a non-win here).

**Measurement caveat (13900HX):** clean nps needs a pinned P-core AND an idle box — the SPRT contending + P/E
scheduling made A/B swings of ±30% (a core frequency dip lands ~62% of turbo, hitting both binaries). Use
fixed-DEPTH time (identical tree for a bit-exact change → compare ms), best-of-N, `taskset -c <P-core>`.

**Next speed lever (biggest ceiling): make/unmake with an accumulator stack** — eliminates the per-node Position
copy entirely (DESIGN.md's aspirational "undo stack"). Large refactor touching every recursion site. But note
the make/unmake result above: reverse-delta unmake was SLOWER; an accumulator-stack variant is the untried form.

**How to apply:** when adding a per-move operation, ask "does this pay a make_move (2KB copy) it could avoid?"
Prune/filter before make_move wherever possible. Any change that alters the search tree must keep the bench node
signature meaningful (a pure-speed change keeps it IDENTICAL — a strong correctness check). See [[zenith-eval-experiments]].
