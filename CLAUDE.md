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
make            # -> ./build/zenith   (clang -std=c17, -O3 -flto -march=native, whole-program single-shot compile)
make ARCH=x86-64-v2   # portable release build (override the default -march=native; see release workflow)
make debug      # -> ./build/zenith-debug  (ASan+UBSan, -O1) — use for any movegen/make_move correctness work
make pext       # -> ./build/zenith-pext  (BMI2 PEXT sliding attacks; bit-identical, ~2% faster on Haswell+/Zen3+)
make datagen    # -> ./build/zenith-datagen + ./build/zenith-bullet2text  (standalone NNUE training-data tools)
make datagen-debug  # -> ASan+UBSan builds of both datagen tools (outside src/, so `make debug` skips them)
make clean
./build/zenith        # interactive UCI loop

make check             # run EVERY gate below (perft, bench-signature, legalcheck, seecheck, fuzzcheck,
                       #   bookcheck, nnuecheck) — mirrors CI; any failure aborts non-zero
./build/zenith perft         # movegen vs known counts (canonical + edge-case catchers + 128-position Ethereal) — PASS
./build/zenith bench [depth] # fixed-depth node signature + nps (default depth 13); guards search determinism
./build/zenith legalcheck    # fast legality/check predicates == copy-make ground truth (differential)
./build/zenith seecheck      # static_exchange_eval vs hand-verified capture positions
./build/zenith fuzzcheck     # malformed-FEN/UCI hardening (memory-safety gate; run under `make debug` for ASan)
./build/zenith nnuecheck <net># incremental accumulator == full refresh, bit-identical
./build/zenith bookcheck     # Polyglot keys vs the 9 official spec vectors
make baseline          # snapshot ./build/zenith -> ./build/zenith-base
tools/sprt.sh ./build/zenith ./build/zenith-base   # self-play SPRT of a change vs the baseline
make tables            # regenerate src/generated/*.inc (Zobrist/ln/geometry/magic constants) — deliberate
make format            # clang-format all sources in place
make hooks             # enable the clang-format pre-commit hook (once per clone; core.hooksPath -> .githooks)
make get-book          # download a free Polyglot book -> books/ (gitignored); print the setoption lines
```

`main` is protected (ruleset: PR required, all four CI checks green, no force pushes; the release
version-bump deploy key is the only bypass). Land changes via branch → `gh pr create` →
`gh pr merge --auto`; direct pushes to main are rejected.

A versioned pre-commit hook (`.githooks/pre-commit`, enabled via `make hooks`) rejects commits whose staged
`src/*.{c,h}` are not clang-format-clean — always keep sources formatted (run `make format`).

The build is **one `clang` invocation over all of `src/*.c`** so LTO sees everything — there are no
object files or per-file targets. `-march=native` is dev-only; a real release fans out per microarch.
`tools/sprt.sh` runs **fastchess** (cutechess-cli is not installed here) with an openings EPD (`$OPENINGS`, default
`~/pawnstar_nnue/openings.epd`); tune via env vars `TC` / `ELO0` / `ELO1` / `CONCURRENCY`.

## The testing discipline (this is the point of the project)

No strength change lands on intuition. Every change to search/eval must:
1. keep `./build/zenith perft` passing (movegen invariant — never regress this),
2. produce a new, deterministic `./build/zenith bench` node signature, and
3. pass a self-play **SPRT** vs the prior baseline (`tools/sprt.sh`).

When you touch search or eval, expect to run all three. Perft and bench are fast local gates; SPRT is the
real verdict on whether a change gains Elo.

## Architecture

Single translation unit per file, flat `src/`. Threads and the monotonic clock go through `platform.h`
(C11 `threads.h` with a pthread fallback for Apple/`__STDC_NO_THREADS__`).

**No globals.** All mutable engine state lives in an `Engine` aggregate (`engine.h`: the TT, a
`SearchShared` with the stop flag/params/LMR table, the eval cache, and the loaded NNUE net), created by
`engine_new()` / destroyed by `engine_delete()` in each executable's `main` and passed explicitly
(`uci_loop`/`run_bench`/`run_datagen` take `Engine *`; every `Searcher` carries `engine`). UCI session state (game, options, searcher pool, book) is a `UciSession` on
`uci_loop`'s stack. The Zobrist keys, Q28 ln table, and the leaper/geometry attack tables
(pawn/knight/king, BetweenBB/LineBB) are generated compile-time constants (`src/generated/*.inc`, from
`tools/generate_tables.py`, which also emits the 128 magic multipliers). The ONE exception:
`bitboard.c`'s ~850KB magic sliding-attack tables — deterministically filled from the constant multipliers
by `init_bitboards()` (idempotent, called lazily by `engine_new()`; no search, no PRNG) before any thread exists, read-only
after; kept file-scope because they sit on the hottest loads (see `bitboard.h`).

- **types.h** — `Color` (with `enemy_of`) and `Piece` — the one piece type, `NO_PIECE=0`, `PAWN=1` … `KING=6`
  (the mailbox stores it directly; color comes from the `colors` bitboards via `position_color_on`). Packed
  16-bit `Move` (from|to|flag, CPW flag encoding), value scale (`VALUE_MATE=32000`, `MAX_PLY=128`), and all
  the `<bit>`-based bitboard helpers (`lsb`/`pop_lsb`/per-direction `shift`/file+rank masks).
- **bitboard.\*** — generated-const pawn/knight/king attacks + `BetweenBB`/`LineBB`; sliding attacks via
  **magic bitboards** (`bishop_attacks`/`rook_attacks`) whose multipliers are generated offline and whose
  tables are filled deterministically at startup. Portable. A `ZENITH_USE_PEXT` compile
  switch (`make pext`) swaps the magic multiply-shift index for a BMI2 `_pext_u64` — bit-identical output
  (same bench signature), ~2% faster perft on Intel Haswell+/AMD Zen3+, but microcoded-slow on AMD Zen1/2,
  so magic stays the portable default and PEXT is opt-in for the fast-BMI2 microarch release variants.
- **position.\*** — board = `colors[2]` + `pieces[NUM_PIECES=7]` bitboards (indexed by `Piece`; the
  `pieces[NO_PIECE]` slot holds the incrementally-maintained occupied-squares bitboard, so
  `position_occupied` is one load) **plus** a `board[64]` piece-type mailbox, kept in sync. Incremental
  **Zobrist** `key` (`ZobristPiece[color][piece][sq]`, `ZobristEp[sq]`) + pawn-only `pawn_key`; the NNUE
  accumulator is embedded and updated
  in the add_piece/remove_piece/move_piece primitives. Copy-free oracles for the search: `position_is_move_legal` (checkers+pins;
  the copy-make `position_is_move_legal_slow` is the reference oracle it is validated against),
  `pinned_to_king`, `gives_check_fast`/`discovered_check_candidates` — all differentially validated by
  `legalcheck`. FEN I/O.
- **movegen.\*** — one masked setwise skeleton (`generate_moves`) instantiated twice: `generate_pseudo`
  (permissive masks — plain pseudo-legal emission into a caller-provided `Move[MAX_MOVES]` buffer,
  `MOVE_NONE`-terminated, returning the count) and `generate_legal` (real check-evasion + pin-ray masks baked
  into the target sets, per-destination king safety, full test only for en passant — no per-move filter pass;
  ~550 Mnps perft, at parity with pawnstar). The search uses `generate_pseudo` and filters with
  `position_is_move_legal` *before* pruning/make (the +66 Elo prune-before-make change); `generate_legal` serves
  perft/datagen/UCI parsing. `noisy_only` = captures+promotions.
- **eval.\*** — `evaluate(pos, cache)` returns centipawns from side-to-move POV (memoised in the engine's
  shared lockless `EvalCache`; NULL = uncached). Always the NNUE forward: every `Engine` holds a net — the
  **build-time-embedded** one (tools/embed_net.py -> build/embedded_net.o, from `NET` in the Makefile) by
  default, replaced by UCI `EvalFile` (an unloadable file falls back to the embedded net). The hand-crafted
  PeSTO fallback was removed with the embedded default; git history keeps it. This single call site is the
  NNUE seam.
- **nnue.\*** — quantised **king-bucketed** (768×8 → 512 → 8 output heads) SCReLU perspective net: loader
  (`ZNNUE5` magic; still reads `ZNNUE4`/`ZNNUE3` so older nets keep working), feature indexing, and the
  integer forward. A perspective whose own king sits on files e–h is **mirrored horizontally** (`square ^ 7`)
  onto files a–d before indexing; the (mirrored) king square then selects one of **8 king-input buckets**
  (4 files × 2 board-halves) offsetting its 768 block, and total piece count selects one of 8 **material
  output buckets**. The accumulator is maintained **incrementally** (embedded in `Position`); a king move
  that changes a side's bucket *or* mirror state triggers `refresh_perspective`, accelerated by a per-thread
  **finny refresh cache** (owned by each `Searcher`, bound into its root position; per
  (perspective,mirror,bucket) cached accumulator + the board it was built from; rebuild applies only
  piece-diffs, per-entry net-pointer-guarded). The accumulator carries its net binding
  (`position_init(pos, net)`; NULL only for never-evaluated positions) — the net itself is heap-loaded and owned by the `Engine`.
  `nnueeval <net>` CLI reads FENs from stdin and prints evals (used by the verification gate).
- **datagen/** (top-level, outside `src/` and the engine binary) — the NNUE training-data tools, built by
  `make datagen` into two standalone executables (each has its own small `main`; both compile the engine core
  `src/*.c` minus `src/main.c` in the same single-shot whole-program style):
  `./build/zenith-datagen <games> <out> [seed] [nodes] [openingPlies]` self-plays from random openings and
  emits `fen;stm_score_cp;wdl` records (one per quiet position; fan out with `tools/datagen_parallel.sh`),
  and `./build/zenith-bullet2text <in.data> <out.txt> [maxRecords] [stride]` converts bulletformat binpacks
  to the same text.
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
- **uci.\*** — protocol loop + the CLI subcommands (`bench`/`perft`/`legalcheck`/`bookcheck`/`nnueeval`/
  `nnuecheck`). Search runs on a coordinator thread (`platform.h` shim); `stop`
  sets the shared atomic `g_stop`. Options: `Hash`, `Clear Hash`, `Threads` (Lazy SMP, 1–256; default = half the logical CPUs, so each thread gets a real core on SMT machines),
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
  king-bucketed 768×8→512→8 output heads, horizontal king mirroring, SCReLU). `src/nnue.c` must reproduce
  `feature_index`, `king_bucket`, `king_mirror`, `output_bucket`, and `integer_eval` **byte-for-byte**. Train two ways: monolithic (`--cache`, ≤~230M positions in RAM) or
  **streaming** (`--shard-dir` of per-shard `.npz` caches, one ~95M shard in RAM at a time — this is how the
  shipped net trained on 650M+ positions). Build shard caches with `--featurise-shard TEXT NPZ` (chunked,
  low-RAM, parallelizable). The current best net `nets/zenith-km1.nnue` = horizontal king mirroring
  (ZNNUE5) + material output buckets + the same 1.4B PlentyChess positions refeaturised under the mirrored
  contract (+14.3 Elo fixed-depth SPRT over ob2, val loss 0.014691 — the project's best). Lineage:
  kb2 (+57, 650M) -> kb3 (+8, 1.4B; data returns diminishing) -> ob2 (+14.7, output buckets) -> km1
  (+14.3, mirroring). `nets/zenith-ob2.nnue` and `nets/zenith-kb3.nnue` are retained for reference.
- **Verification gate (never skip):** `trainer/verify.py` runs `./build/zenith nnueeval` and diffs against the
  Python reference — must be **0 cp** (bit-identical). Also check symmetry: `eval(pos) == eval(color-mirror)`.
- **The trained net is a faithful executor** — if the engine plays badly, suspect the *net/data* (eval
  noise), not the loader. The pilot net loses to HCE because minimax amplifies leaf-eval noise; see the
  `zenith-nnue-pilot-status` memory. Fix = more/cleaner data + better training, not engine code.
- Every net change is still SPRT-gated (`tools/sprt.sh`, fastchess): `CAND_NET`/`BASE_NET` set `EvalFile`
  per side (unset ⇒ that side uses its embedded net). Fixed-depth matches isolate eval quality from NNUE's speed cost.

## Conventions

- Search values are side-to-move-relative (negamax). Mate scores are `±(VALUE_MATE - ply)`; test with
  `is_mate_score`, never bare comparisons.
- Continuation-history / countermove key is `(piece, to-square)` of the move that reached a node, encoded
  `piece*64 + to` (range 768); contHist is indexed `prevPT*768 + curPT`.
- Prefer the existing bit helpers in `types.h` over hand-rolled bit twiddling; sliding attacks always go
  through the magic-bitboard functions, never a raw ray loop.
