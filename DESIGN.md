# Zenith — design & roadmap to a world-class UCI engine

## Goal & honest scope

A world-class engine (≈3500–3650 CCRL) is not primarily a coding problem. The search/eval **code**
that separates a 3000-Elo engine from Stockfish is a few thousand lines; the gap is:

1. an **NNUE** trained on ~10⁹–10¹⁰ positions (self-play or filtered game data), and
2. **thousands of SPRT-gated micro-improvements** validated on a Fishtest-style cluster.

So Zenith is engineered as a *world-class architecture* — every structural piece the top engines have,
with clean seams for the data/compute program that climbs the last ~600 Elo. The plan below has been
executed: the perft-proven classical core, then NNUE (own trainer + king-bucketed net) and Lazy SMP on
top, each step SPRT-gated. Zenith now beats its reference engine (pawnstar) at every tested config.

## Language: C (clean-sheet, independent; originally C++20 — see tag cpp-final)

C17 (ported from C++20), chosen for parity performance with the top engines and zero-overhead control over memory layout and
concurrency — decisive for the lockless TT and Lazy SMP — over managed languages like Go, whose
GC pauses are unacceptable on the search hot path. Zero-cost abstractions, a built-in fixed-depth `bench`
node signature, and a binary-vs-binary SPRT harness. Portable magic bitboards keep the core arch-independent;
release builds are per-microarchitecture (`x86-64-v3` etc.).

## Architecture

```
main.c       entry
uci.{h,c}    protocol parse/format (position/go/setoption/info/bestmove) + time manager
types.h      Color, Piece, Square, Move (uint16 + inline accessors), CastleRights, bit tricks
platform.h   C11 threads.h / pthread shim + monotonic clock
bitboard.{h,c} precomputed attacks; rook/bishop MAGIC bitboards (runtime-generated, seeded)
position.{h,c} bitboard board + mailbox, Zobrist + pawn key, FEN, copy-make, legality/check oracles
movegen.{h,c}  pseudo-legal generation (+ legal wrapper); search filters via the legality oracle
eval.{h,c}     evaluate(): NNUE when a net is loaded, else tapered HCE; shared eval cache
nnue.{h,c}     king-bucketed quantised NNUE + AVX2 forward + finny refresh cache
book.{h,c}     Polyglot opening book
tt.{h,c}       lockless transposition table ({key^data, data} slots, bit-field payload)
search.{h,c}   iterative deepening, PVS, TT, quiescence, pruning/reductions, SPSA-tunable params
```

### Board representation
- Bitboards (`by_color[2]`, `by_type[6]`) + a `[Piece; 64]` mailbox for O(1) lookup.
- **Zobrist** hashing, maintained incrementally in make/unmake.
- **Copy-make**: `Position` is a value type; search copies the parent and applies `make_move` to the copy,
  "undoing" by discarding it (no undo stack, no `unmake_move`). ~2.4 Mnps single-thread with NNUE.
  Make/unmake was tried and measured SLOWER (the accumulator's feature-column reads dominate, not the
  copy) — see `.claude/memory/search-speed-levers.md` before revisiting.
- Sliding attacks via **magic bitboards** generated at startup (portable; PEXT is a drop-in for BMI2 CPUs).

### Search (the classical strength engine)
Fail-soft **PVS** inside iterative deepening with **aspiration windows**. On top:
- **Transposition table**: lockless (atomic 16-byte cells, Hyatt XOR key), depth-preferred replacement,
  bounds (exact/lower/upper), TT move drives ordering.
- **Move ordering**: TT move → winning captures (SEE / MVV-LVA) → killers → history (butterfly +
  continuation) → quiets → losing captures.
- **Quiescence**: captures + queen promotions, SEE-pruned, stand-pat.
- **Pruning/reductions** (implemented): null-move (adaptive R), reverse futility, futility, late-move
  pruning, late-move reductions (log table), SEE pruning, IIR, and a pawn-keyed eval correction history.
  (Razoring/ProbCut/delta pruning remain unimplemented candidates; history-LMR tested neutral.)
- **Extensions**: check, singular (TT-move exclusion search).
- Mate-distance pruning, repetition/50-move draw detection.
- **Lazy SMP** for multi-thread scaling (shared TT; per-thread everything else).

### Evaluation
A single `int evaluate(const Position&)` seam (centipawns, side-to-move POV) — the one call site NNUE replaces.
- **Fallback:** PeSTO tapered mg/eg material + PST, plus bishop pair, mobility, and tempo.
- **Primary (shipped):** **NNUE** — king-bucketed 768×8→512 perspective net, SCReLU, int16-quantised,
  incremental accumulator + finny refresh cache — behind the same `evaluate()` seam; trained with the
  repo's own PyTorch pipeline (NNUE_TRAINING.md).

## Roadmap (how the last 600 Elo is actually earned)

| Phase | Deliverable | Verification | Status |
|--|--|--|--|
| **0 Board** | magic movegen, copy-make, Zobrist | **perft** == known counts (startpos, Kiwipete, CPW 3–6) | ✅ done |
| **1 Search** | PVS+TT+qsearch+ordering+core pruning, UCI, time mgmt | plays legal games; `bench` signature; mate suites | ✅ done |
| **2 Strength** | full pruning/reduction stack, SEE, singular, LMR table | **SPRT** each change (self-play, [0,5]/[−3,1] bounds) | ✅ done |
| **3 Parallel** | Lazy SMP + lockless TT | ThreadSanitizer-clean; N-thread speedup | ✅ done (+203 Elo @ 8 threads) |
| **4 NNUE** | trainer (PyTorch) + data-gen (self-play) + incremental infer | eval MSE vs search; **SPRT** vs HCE (expect +500–700) | ✅ done (king-bucketed kb3) |
| **5 Tuning** | SPSA/Texel tuning of search params + eval; opening-book; syzygy TB | Fishtest-style gauntlets vs reference engines | 🔶 SPSA + book done; syzygy open |

**Testing is the product.** A binary-vs-binary SPRT harness (fastchess, via `tools/sprt.sh`) gates *every*
change; no change lands on strength intuition. Fixed-depth `bench` node signature guards determinism; perft
guards movegen; `legalcheck`/`nnuecheck`/`bookcheck` differentially validate the fast paths against ground
truth. Phases 0–4 of the roadmap are complete (plus SPSA tuning and the opening book from Phase 5);
syzygy tablebases remain open.

## Remaining non-goals
Syzygy tablebases and a Fishtest-style distributed test cluster. Everything else on this page — NNUE
weights, the tuned parameter set, the opening book, Lazy SMP — is done and SPRT-gated.
