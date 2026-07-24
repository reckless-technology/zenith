# Zenith — design & roadmap to a world-class UCI engine

## Goal & honest scope

A world-class engine (≈3500–3650 CCRL) is not primarily a coding problem. The search/eval **code**
that separates a 3000-Elo engine from Stockfish is a few thousand lines; the gap is:

1. an **NNUE** trained on ~10⁹–10¹⁰ positions (self-play or filtered game data), and
2. **thousands of SPRT-gated micro-improvements** validated on a Fishtest-style cluster.

So Zenith is engineered as a *world-class architecture* — every structural piece the top engines have,
with clean seams for the data/compute program that climbs the last ~600 Elo. This repo delivers a
**correct, strong, single-binary classical core** first (perft-proven, modern search, tapered HCE),
then layers NNUE and parallel search on top.

## Language: C (clean-sheet, independent; originally C++20 — see tag cpp-final)

C17 (ported from C++20), chosen for parity performance with the top engines and zero-overhead control over memory layout and
concurrency — decisive for the (planned) lockless TT and Lazy SMP — over managed languages like Go, whose
GC pauses are unacceptable on the search hot path. Zero-cost abstractions, a built-in fixed-depth `bench`
node signature, and a binary-vs-binary SPRT harness. Portable magic bitboards keep the core arch-independent;
release builds are per-microarchitecture (`x86-64-v3` etc.).

## Architecture

```
main.c       entry
uci.{h,cpp}  protocol parse/format (position/go/setoption/info/bestmove) + time manager
types.h      Color, Piece, Square, Move (uint16), CastleRights, bit tricks (<bit>)
bitboard.{h,cpp} precomputed attacks; rook/bishop MAGIC bitboards (runtime-generated, seeded)
position.{h,cpp} bitboard board + mailbox, Zobrist, FEN, copy-make, attacks-to / in-check
movegen.{h,cpp}  pseudo-legal generation + make/legality filter (perft-gated); staged ordering later
eval.{h,cpp}     evaluate(): tapered HCE now, NNUE later (same call site)
tt.{h,cpp}       transposition table (depth-preferred, bounds); lockless atomic cells arrive with Lazy SMP (Phase 3)
search.{h,cpp}   iterative deepening, PVS, TT, quiescence, pruning/reductions
```

### Board representation
- Bitboards (`by_color[2]`, `by_type[6]`) + a `[Piece; 64]` mailbox for O(1) lookup.
- **Zobrist** hashing, maintained incrementally in make/unmake.
- **Copy-make**: `Position` is a value type; search copies the parent and applies `make_move` to the copy,
  "undoing" by discarding it (no undo stack, no `unmake_move`). ~1.8 Mnps today. A copy-free make/unmake
  with an undo stack is a possible later optimization for the deep alpha-beta hot path.
- Sliding attacks via **magic bitboards** generated at startup (portable; PEXT is a drop-in for BMI2 CPUs).

### Search (the classical strength engine)
Fail-soft **PVS** inside iterative deepening with **aspiration windows**. On top:
- **Transposition table**: lockless (atomic 16-byte cells, Hyatt XOR key), depth-preferred replacement,
  bounds (exact/lower/upper), TT move drives ordering.
- **Move ordering**: TT move → winning captures (SEE / MVV-LVA) → killers → history (butterfly +
  continuation) → quiets → losing captures.
- **Quiescence**: captures + queen promotions, SEE-pruned, stand-pat.
- **Pruning/reductions**: null-move (adaptive R), reverse futility / static null move, futility,
  late-move pruning, late-move reductions (log table, history-adjusted), SEE pruning, delta pruning,
  razoring, ProbCut.
- **Extensions**: check, singular (TT-move exclusion search).
- Mate-distance pruning, repetition/50-move draw detection.
- **Lazy SMP** for multi-thread scaling (shared TT; per-thread everything else).

### Evaluation
A single `int evaluate(const Position&)` seam (centipawns, side-to-move POV) — the one call site NNUE replaces.
- **v1 (now):** PeSTO tapered mg/eg material + PST, plus bishop pair, mobility, and tempo. (King safety,
  pawn structure, rooks on open/semi-open files, and threats are HCE terms to add before NNUE lands.)
- **v2:** **NNUE** (768→N perspective net, king-bucketed, SCReLU, int16/int8 quantized, incremental
  accumulator) — the same `Evaluator` seam; trained via the pipeline below.

## Roadmap (how the last 600 Elo is actually earned)

| Phase | Deliverable | Verification |
|--|--|--|
| **0 Board** | magic movegen, copy-make, Zobrist | **perft** == known counts (startpos, Kiwipete, CPW 3–6) |
| **1 Search** | PVS+TT+qsearch+ordering+core pruning, UCI, time mgmt | plays legal games; `bench` signature; mate suites |
| **2 Strength** | full pruning/reduction stack, SEE, singular, LMR table | **SPRT** each change (self-play, [0,5]/[−3,1] bounds) |
| **3 Parallel** | Lazy SMP + lockless TT | ThreadSanitizer-clean; N-thread speedup |
| **4 NNUE** | trainer (PyTorch) + data-gen (self-play) + incremental infer | eval MSE vs search; **SPRT** vs HCE (expect +500–700) |
| **5 Tuning** | SPSA/Texel tuning of search params + eval; opening-book; syzygy TB | Fishtest-style gauntlets vs reference engines |

**Testing is the product.** A binary-vs-binary SPRT harness (`cutechess-cli`, via `tools/sprt.sh`) gates *every* change; no change
lands on strength intuition. Fixed-depth `bench` node signature guards determinism; perft guards movegen;
a mate-in-N suite guards search correctness.

## Non-goals for v1
NNUE weights (needs training compute), syzygy tablebases, and the tuned parameter set — all are seams,
not blockers. The v1 core is a complete, correct, ~2800–3100-class classical engine.
