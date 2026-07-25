# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Zenith is a from-scratch UCI chess engine in **C17** (~5,300 lines, `src/*.{h,c}`; ported from the original
C++20, preserved at git tag `cpp-final`). It is feature-complete: king-bucketed NNUE, Lazy SMP, an
SPSA-tuned search, and a Polyglot opening book — and it beats its reference engine (pawnstar) at every
tested configuration. Read [README.md](README.md) for the architecture and
[NNUE_TRAINING.md](NNUE_TRAINING.md) for the training pipeline.

## Build / test / run

```bash
make            # -> ./zenith   (clang -std=c17, -O3 -flto -march=native, whole-program single-shot compile)
make debug      # -> ./zenith-debug  (ASan+UBSan, -O1) — use for any movegen/make_move correctness work
make clean
./zenith        # interactive UCI loop

make check             # run EVERY gate below (perft, bench-signature, legalcheck, seecheck, fuzzcheck,
                       #   bookcheck, nnuecheck) — mirrors CI; any failure aborts non-zero
./zenith perft         # movegen vs known counts (canonical + edge-case catchers + 128-position Ethereal) — PASS
./zenith bench [depth] # fixed-depth node signature + nps (default depth 13); guards search determinism
./zenith legalcheck    # fast legality/check predicates == copy-make ground truth (differential)
./zenith seecheck      # static_exchange_eval vs hand-verified capture positions
./zenith fuzzcheck     # malformed-FEN/UCI hardening (memory-safety gate; run under `make debug` for ASan)
./zenith nnuecheck <net># incremental accumulator == full refresh, bit-identical
./zenith bookcheck     # Polyglot keys vs the 9 official spec vectors
make baseline          # snapshot ./zenith -> ./zenith-base
tools/sprt.sh ./zenith ./zenith-base   # self-play SPRT of a change vs the baseline
make format            # clang-format all sources in place
make hooks             # enable the clang-format pre-commit hook (once per clone; core.hooksPath -> .githooks)
```

A versioned pre-commit hook (`.githooks/pre-commit`, enabled via `make hooks`) rejects commits whose staged
`src/*.{c,h}` are not clang-format-clean — always keep sources formatted (run `make format`).

The build is **one `clang` invocation over all of `src/*.c`** so LTO sees everything — there are no
object files or per-file targets. `-march=native` is dev-only; a real release fans out per microarch.
`tools/sprt.sh` runs **fastchess** (cutechess-cli is not installed here) with an openings EPD (`$OPENINGS`, default
`~/pawnstar_nnue/openings.epd`); tune via env vars `TC` / `ELO0` / `ELO1` / `CONCURRENCY`.

## The testing discipline (this is the point of the project)

No strength change lands on intuition. Every change to search/eval must:
1. keep `./zenith perft` passing (movegen invariant — never regress this),
2. produce a new, deterministic `./zenith bench` node signature, and
3. pass a self-play **SPRT** vs the prior baseline (`tools/sprt.sh`).

When you touch search or eval, expect to run all three. Perft and bench are fast local gates; SPRT is the
real verdict on whether a change gains Elo.

## Architecture

Single translation unit per file, flat `src/`. Threads and the monotonic clock go through `platform.h`
(C11 `threads.h` with a pthread fallback for Apple/`__STDC_NO_THREADS__`). Init order in `main.c` matters:
`init_bitboards()` → `init_zobrist()` → `init_eval()` → `init_search()` → `TT.resize()`.

- **types.h** — `Color` (with `enemy_of`) and `Piece` — the one piece type, `NO_PIECE=0`, `PAWN=1` … `KING=6`
  (the mailbox stores it directly; colour comes from the by_color bitboards via `position_color_on`). Packed
  16-bit `Move` (from|to|flag, CPW flag encoding), value scale (`VALUE_MATE=32000`, `MAX_PLY=128`), and all
  the `<bit>`-based bitboard helpers (`lsb`/`pop_lsb`/per-direction `shift`/file+rank masks).
- **bitboard.\*** — precomputed pawn/knight/king attacks + `BetweenBB`; sliding attacks via **magic
  bitboards generated at startup** (`bishop_attacks`/`rook_attacks`). Portable; PEXT is a drop-in later.
- **position.\*** — board = `by_color[2]` + `by_type[PIECE_NB=7]` bitboards (indexed by `Piece`; the
  `by_type[NO_PIECE]` slot holds the incrementally-maintained occupied-squares bitboard, so
  `position_occupied` is one load) **plus** a `board[64]` piece-type mailbox, kept in sync. Incremental
  **Zobrist** `key` (`ZobristPiece[color][piece][sq]`, `ZobristEp[sq]`) + pawn-only `pawn_key`; the NNUE
  accumulator is embedded and updated
  in the put/remove/move primitives. Copy-free oracles for the search: `is_legal_fast` (checkers+pins),
  `pinned_to_king`, `gives_check_fast`/`discovered_check_candidates` — all differentially validated by
  `legalcheck`. FEN I/O.
- **movegen.\*** — `generate_pseudo` emits pseudo-legal moves into a fixed `MoveList` (`Move moves[256]`);
  the search filters with `is_legal_fast` *before* pruning/make (the +66 Elo prune-before-make change).
  `generate_legal` (pseudo + filter) serves perft/datagen/UCI parsing. `noisy_only` = captures+promotions.
- **eval.\*** — `evaluate(pos)` returns centipawns from side-to-move POV. Returns `nnue::evaluate(pos)`
  when a net is loaded (UCI `EvalFile`), else the PeSTO tapered HCE (material+PST, bishop pair, mobility,
  tempo). This single call site is the NNUE seam.
- **nnue.\*** — quantised **king-bucketed** (768×8 → 512) SCReLU perspective net: loader (`ZNNUE3` magic),
  feature indexing, and the integer forward. The perspective's own king square selects one of **8 king-input
  buckets** (4 file-pairs × 2 board-halves) offsetting its 768 block. The accumulator is maintained
  **incrementally** (embedded in `Position`); a king move that changes a side's bucket triggers
  `refresh_perspective`, accelerated by a thread-local **finny refresh cache** (per (perspective,bucket)
  cached accumulator + the board it was built from; rebuild applies only piece-diffs, net-generation-guarded).
  `nnueeval <net>` CLI reads FENs from stdin and prints evals (used by the verification gate).
- **datagen.\*** — `datagen <games> <out> [seed] [nodes] [openingPlies]` self-plays from random openings and
  emits `fen;stm_score_cp;wdl` records (one per quiet position). Fan out with `tools/datagen_parallel.sh`.
- **tt.\*** — global, **lockless** for Lazy SMP: 16-byte `{key^data, data}` slots with the XOR torn-read
  guard, relaxed atomics, bit-field payload (`TTData`, signed depth), depth-preferred replacement with
  generation aging. Mate scores are stored distance-from-node: **always go through
  `score_to_tt`/`score_from_tt`** across the TT boundary.
- **search.\*** — the strength engine, all in `Searcher`: iterative deepening + aspiration windows,
  fail-soft PVS `negamax`, `qsearch` (SEE-pruned), ordering (TT move → MVV-LVA → killers → countermove →
  butterfly+continuation history), pruning/reductions (null-move, reverse-futility, futility, LMP, LMR,
  SEE pruning, IIR), check + singular extensions, mate-distance pruning, and a pawn-keyed eval
  **correction history**. All margins live in `SearchParams` (UCI-exposed spins, **SPSA-tuned** defaults;
  re-tune with `tools/spsa.py`). `static_exchange_eval()` is the local SEE.
- **uci.\*** — protocol loop + the CLI subcommands (`bench`/`perft`/`legalcheck`/`bookcheck`/`datagen`/
  `bullet2text`/`nnueeval`/`nnuecheck`). Search runs on a coordinator thread (`platform.h` shim); `stop`
  sets the shared atomic `g_stop`. Options: `Hash`, `Clear Hash`, `Threads` (Lazy SMP, 1–256),
  `Move Overhead`, `EvalFile`, `OwnBook`/`BookFile` (Polyglot; OwnBook defaults false — testing stays
  bookless), plus the SPSA-tunable search parameters. Prints a version banner (`src/version.h`:
  major.minor.<git commit count>, stamped by the Makefile).

### Copy-make, not an undo stack

`Position` is a value type; search does `Position child = pos; child.make_move(m);` and "undoes" by
discarding the copy (see `position.h` and every recursion site in `search.c`). There is **no**
`unmake_move`. Adding one is a real architectural change, not a bug fix — and it was tried and measured
*slower* (the embedded accumulator's feature-column reads dominate; see the `search-speed-levers` memory).

### Repetition / draw history

Draw detection needs positions played *before* the search root. `uci.c` accumulates pre-root keys in
`game_hist` (rebuilt on each `position` command) and hands them to each searcher's `hist_keys`. Inside `negamax`, the
current `pos.key` is `push_back`/`pop_back`-ed around each recursive call so `is_draw()` can scan back to
the last irreversible move. If you add a make/recurse site, maintain this `hist` push/pop or repetition
detection breaks.

## NNUE training pipeline (independent — no shared code/net/data with any other engine)

`datagen` (C) → `trainer/train.py` (PyTorch, CUDA) → quantised `.nnue` → `src/nnue.c` (engine). The
venv is `.venv` (torch + numpy, gitignored); `data/` and `nets/` are gitignored.

- **The contract is `trainer/features.py`** — feature indexing + quantisation (QA=255, QB=64, scale=400,
  king-bucketed 768×8→512, SCReLU). `src/nnue.c` must reproduce `feature_index`, `king_bucket`, and
  `integer_eval` **byte-for-byte**. Train two ways: monolithic (`--cache`, ≤~230M positions in RAM) or
  **streaming** (`--shard-dir` of per-shard `.npz` caches, one ~95M shard in RAM at a time — this is how the
  shipped net trained on 650M+ positions). Build shard caches with `--featurise-shard TEXT NPZ` (chunked,
  low-RAM, parallelizable). The current best net `nets/zenith-kb3.nnue` = king buckets + 1.4B PlentyChess
  positions (16 shards); +57 Elo (kb2, 650M) over the prior 512/190M net, then +8 more (kb3) — data returns
  are now diminishing.
- **Verification gate (never skip):** `trainer/verify.py` runs `./zenith nnueeval` and diffs against the
  Python reference — must be **0 cp** (bit-identical). Also check symmetry: `eval(pos) == eval(color-mirror)`.
- **The trained net is a faithful executor** — if the engine plays badly, suspect the *net/data* (eval
  noise), not the loader. The pilot net loses to HCE because minimax amplifies leaf-eval noise; see the
  `zenith-nnue-pilot-status` memory. Fix = more/cleaner data + better training, not engine code.
- Every net change is still SPRT-gated (`tools/sprt.sh`, fastchess): `CAND_NET`/`BASE_NET` set `EvalFile`
  per side (unset ⇒ that side uses HCE). Fixed-depth matches isolate eval quality from NNUE's speed cost.

## Conventions

- Search values are side-to-move-relative (negamax). Mate scores are `±(VALUE_MATE - ply)`; test with
  `is_mate_score`, never bare comparisons.
- Continuation-history / countermove key is `(piece, to-square)` of the move that reached a node, encoded
  `piece*64 + to` (range 768); contHist is indexed `prevPT*768 + curPT`.
- Prefer the existing bit helpers in `types.h` over hand-rolled bit twiddling; sliding attacks always go
  through the magic-bitboard functions, never a raw ray loop.
