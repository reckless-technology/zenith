---
name: search-speed-levers
description: "What actually moves Zenith's nps/strength — copy-make cost is the dominant per-node lever"
metadata: 
  node_type: memory
  type: project
  originSessionId: 221925ef-3c7d-4275-96ac-0cd7362cb511
  modified: 2026-07-26T04:16:37.029Z
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
- **Lazy / deferred accumulator (dirty-piece) was tried and REVERTED (negative result, 2026-07-25): ~19% SLOWER
  with the net loaded** (2.65M→2.15M nps, identical node count; nnuecheck 0-mismatch + identical NNUE search
  node counts confirmed it was bit-exact). Recording feature deltas in a pending list and applying them only at
  eval time saves the FT reads only for nodes that never eval — but Zenith's search computes a static eval at
  *most* nodes (RFP/futility/NMP/qsearch stand-pat), so they materialise anyway, and the per-node overhead
  (delta recording + a bigger copy-make + the materialise indirection) hits EVERY node. Same root cause as
  make/unmake: the FT reads are unavoidable once you eval, and nearly every node evals. Don't retry unless a
  large fraction of nodes can be made to skip eval.

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

- **TT prefetch: pure-speed win** (2026-07-25, commit 0aefd74). `tt_prefetch(key)` = `__builtin_prefetch(&TT.table[key & mask])`, called right after each `make_move` in negamax/qsearch so the slot loads while the node finishes (extensions, hist push, reduction) before the recursive probe. Bench signature 3065743 unchanged (pure speed); ~+4-7% bench nps. Cheap, low-risk latency hiding on the memory-bound lockless TT.

- **AVX-512 forward-pass variant: tried, then REMOVED (commit f46bcac added it, reverted right after,
  2026-07-25).** Widening the NNUE column add/sub + SCReLU dot 256->512-bit (`__AVX512BW__`, -march=x86-64-v4)
  is bit-identical to AVX2 by construction, but there is NO way to run/validate/measure it on this setup: the
  13900HX has AVX-512 fused off (cpuinfo: only avx2 + avx_vnni), qemu-user 8.2's TCG doesn't emulate AVX-512
  (a bare `vpminsw zmm` SIGILLs even with `-cpu max`, despite -cpu help listing the flag), and no Intel SDE.
  An unrunnable code path earns nothing in-tree, so it was removed — revisit only with real AVX-512 silicon or
  a v4 CI runner (validate via nnuecheck 0-mismatch + node-count match vs AVX2). Also: **AVX-VNNI (present on
  this CPU) can't accelerate this net** — VNNI is int16xint16->int32 madd, but SCReLU is clamped^2 * weight
  (int32 intermediates); VNNI would only help an int8-quantised net (a trainer/format change).

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
copy entirely (the aspirational "undo stack"). Large refactor touching every recursion site. But note
the make/unmake result above: reverse-delta unmake was SLOWER; an accumulator-stack variant is the untried form.

**How to apply:** when adding a per-move operation, ask "does this pay a make_move (2KB copy) it could avoid?"
Prune/filter before make_move wherever possible. Any change that alters the search tree must keep the bench node
signature meaningful (a pure-speed change keeps it IDENTICAL — a strong correctness check). See [[zenith-eval-experiments]].

**Perft/movegen parity with pawnstar (2026-07-25, commits 49fb310/459b653/bcfb413).** Investigated "pawnstar
perft ~10x faster": root cause was generate_legal filtering every pseudo move with the copy-make
position_is_legal (a 2.3KB copy + make_move PER CANDIDATE MOVE) — search had the fast oracle for months but
perft/datagen/UCI never got it. Fix chain, each verified count-identical + bench 3065743: (1) copy-free
filter in generate_legal: 40 -> 213 Mnps; (2) perft_copy skips the dead accumulator: -> 233; (3) mask-based
legal generation (one masked setwise skeleton, generate_pseudo = permissive masks so search emissions are
byte-identical; check-evasion + pin-ray masks, king-safety per destination, full test only for ep): -> ~505
mean, startpos-d6 554 vs pawnstar 534-545 (parity; pawnstar's remaining edge was template monomorphization —
not needed). Ablations along the way: zobrist updates ~5%, checkers recompute ~noise; once the filter is copy-free the
recursion copy is a minor cost (perft_copy +10%). Lesson: attribute
before optimizing — the "obvious" accumulator-copy theory measured +10%, the real cost was an O(make_move)
legality test per generated move. PROFILE-CONFIRMED after the user set perf_event_paranoid=1: old binary = 44.5% libc memmove (the 2.3KB copy PER LEGALITY TEST, ~35x per node) + 33.8% position_make_move (double-make) = ~78% overhead, generate_pseudo only 6%; new binary = 69% generate_moves, 11% make_move, memmove gone. The early "recursion-copy-skip barely helps" confusion: it ablated 1 of ~36 copies per node — the hot copies were inside position_is_legal. (perf works now; on this hybrid CPU perf report prints one section per core-type PMU.)
