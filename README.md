# Zenith

A UCI chess engine in **C17** with its own independently-trained NNUE evaluation. 

The source code was written by Claude Code under the close supervision and direction of Jonny Reckless.

Zenith pairs a modern alpha-beta search — iterative deepening, principal-variation search, a lockless
transposition table, the full pruning/reduction stack, and Lazy SMP — with a king-bucketed neural
network evaluation trained by the repository's own pipeline.

Goals for the project were:
- Keep the source code relatively small and simple.
- Use C for portability and speed.
- Use an NNUE based evaluator.
- Train the evaluation weights as part of the project.
- Competitive play strength (greater than or equal to pawnstar).

## Build

Requires a C17 compiler (clang 18 recommended; gcc 13+ builds warning-clean and produces the identical
bench node signature). Threads use C11 `threads.h` with a
pthread fallback (`src/platform.h`) for platforms without it (e.g. macOS). Magic bitboards keep the code
portable; `-march=native` is for local dev — a release build would fan out per microarchitecture.

```bash
make                 # -> ./build/zenith        (release: clang -std=c17, -O3 -flto -march=native, whole-program LTO)
make ARCH=x86-64-v2  # portable release build: override the default -march=native (see .github/workflows/release.yml)
make debug           # -> ./build/zenith-debug  (AddressSanitizer + UBSan, -O1 -g; use for movegen/make_move work)
make pext            # -> ./build/zenith-pext   (BMI2 PEXT sliding attacks; bit-identical, ~2% faster on Haswell+/Zen3+)
make check           # build + run every correctness gate (mirrors CI; see Verify below)
make perft           # build + run the perft movegen gate
make bench           # build + run the fixed-depth node-signature benchmark
make baseline        # snapshot the current ./build/zenith -> ./build/zenith-base (the SPRT reference binary)
make get-book        # download a free Polyglot opening book -> books/ (gitignored); prints the setoption lines
make tables          # regenerate the committed constant tables (src/generated/*.inc: Zobrist, PeSTO,
                     #   Q28 ln, bitboard geometry, magic multipliers) — a
                     #   deliberate step, never a build side effect; the bench signature guards the values
make format          # clang-format all sources in place (src/*.{c,h})
make hooks           # install the clang-format pre-commit hook (once per clone; core.hooksPath -> .githooks)
make doc             # -> doc/html/index.html (Doxygen API reference)
make clean           # remove build outputs (binaries + doc/html)
./build/zenith       # interactive UCI (prints "Zenith <major>.<minor>.<git commit count>")
```

## Verify

Every correctness property has an executable gate. `make check` runs them all; CI runs the same gates on
Linux, macOS, and Windows (asserting the bench signature on each), plus a threaded UCI smoke and the NNUE
gates against a synthetic net. Any failure aborts non-zero.

```bash
make check                          # build + run every gate below
./build/zenith perft                # movegen vs known counts: canonical + ep/castling/promotion catchers + Ethereal 128
./build/zenith bench 13             # deterministic fixed-depth node signature (guards search behaviour) + nps
./build/zenith legalcheck           # the copy-free legality/check predicates == copy-make ground truth over a perft walk
./build/zenith seecheck             # static exchange evaluation vs hand-verified capture positions
./build/zenith fuzzcheck            # malformed-FEN/UCI hardening (memory safety; run the `make debug` build under ASan)
./build/zenith nnuecheck <net.nnue> # incremental accumulator == full refresh, bit-identical
./build/zenith bookcheck            # Polyglot key computation vs the 9 official spec test vectors
```

The **`bench` node signature** is the linchpin: a fixed-depth search produces a deterministic node count,
so any unintended change to search behaviour shows up immediately, and a pure-speed change is proven safe
by leaving it identical. `tools/sprt.sh` runs a self-play SPRT (via **fastchess**) to decide whether a
change that *does* alter the signature is actually a strength gain — no strength change lands on intuition.

## Architecture

One translation unit per file, flat `src/`. **No globals:** all mutable engine state lives in an `Engine`
aggregate (`engine.h` — the transposition table, the shared search state, the eval cache, and the loaded
NNUE net), built by `engine_new()` and released by `engine_delete()` in `main`, passed explicitly
everywhere; UCI session state is a `UciSession` on `uci_loop`'s stack. The Zobrist keys, PeSTO tables, Q28 ln table, and the leaper/geometry attack tables
are generated compile-time constants (`tools/generate_tables.py` → `src/generated/*.inc`). The one
exception is
`bitboard.c`'s ~850KB magic sliding-attack tables, deterministically filled from generated constant
multipliers by `init_bitboards()` — main's only startup call, with no search or PRNG — before any thread
exists.

| File | Contains |
|---|---|
`src/types.h`      | Color, Piece (NO_PIECE=0..KING=6), Square, packed 16-bit Move + inline accessors, bit helpers
`src/platform.h`   | C11 threads.h / pthread thread shim + a monotonic clock
`src/bitboard.*`   | precomputed pawn/knight/king attacks, BetweenBB/LineBB, rook/bishop MAGIC bitboards
`src/position.*`   | bitboards + mailbox, Zobrist + pawn key, FEN I/O, copy-make, legality/check oracles
`src/movegen.*`    | one masked setwise generator: pseudo-legal + single-pass legal instantiations
`src/engine.*`     | the Engine aggregate (TT + shared search state + eval cache + NNUE net) + its new/delete
`src/eval.*`       | evaluate(): NNUE when a net is loaded, else a tapered HCE; shared eval cache
`src/nnue.*`       | king-bucketed quantised NNUE: loader, feature indexing, AVX2 forward, finny refresh cache
`src/book.*`       | Polyglot opening book: key computation, probing, weighted move choice
`src/tt.*`         | lockless transposition table ({key^data, data} slots, bit-field payload)
`src/search.*`     | iterative deepening, PVS, quiescence, the pruning/reduction/extension stack, Lazy SMP
`src/datagen.*`    | self-play data generation + a bulletformat-to-text converter (for training)
`src/uci.*`        | the UCI protocol loop, time manager, and CLI self-test subcommands
`src/version.h`    | major.minor + build number (git commit count, stamped by the Makefile)
`src/main.c`       | entry + CLI dispatch

### Board representation

- **Bitboards** — `colors[2]` and `pieces[7]` (indexed by `Piece`; the `NO_PIECE` slot holds the
  occupied-squares bitboard) — plus a `board[64]` piece-type mailbox (color comes from `colors`) for
  O(1) piece lookup, all kept in sync by the add_piece/remove_piece/move_piece primitives.
- **Zobrist** hashing maintained incrementally, with a separate pawn-only key used by the eval correction
  history in search.
- **Copy-make.** `Position` is a value type: the search copies the parent and applies `make_move` to the
  copy, "undoing" simply by discarding it — there is no undo stack and no `unmake_move`. The NNUE
  accumulator is embedded in `Position` and updated incrementally by the same primitives, so the copy
  carries a ready-to-use accumulator. (An in-place make/unmake was measured *slower*: the accumulator's
  feature-column reads dominate, and unmake would double them to save a cheap copy.)
- **Sliding attacks** via **magic bitboards**: the multipliers are generated offline (`make tables`) and
  the attack tables filled deterministically at startup — portable, instant init. `make pext`
  (`ZENITH_USE_PEXT`) swaps in a BMI2 `_pext_u64` index instead: bit-identical output, ~2% faster on Intel
  Haswell+/AMD Zen3+, but microcoded-slow on AMD Zen1/2 — so magic is the portable default and PEXT is
  opt-in for fast-BMI2 release builds.

### Move generation

One masked setwise generator (`generate_moves`) is instantiated twice. `generate_pseudo` uses permissive
masks and emits pseudo-legal moves into a caller-provided `Move[MAX_MOVES]` buffer terminated by a
`MOVE_NONE` sentinel; the search then filters each move with the copy-free legality oracle —
`position_is_legal(move, checkers, pinned)`, built on precomputed pins and checkers — *before* it pays for
`make_move`, so illegal and pruned moves never cost a copy. `generate_legal` instantiates the same skeleton
with real check-evasion and pin-ray masks baked into the target sets (per-destination king safety; a full
test only for en passant), producing fully legal moves in a single pass with no filter — this backs perft
(~550 Mnps), datagen, and UCI move parsing. The oracles (`position_is_legal`, `pinned_to_king`,
`gives_check_fast`, `discovered_check_candidates`) are differentially validated against the copy-make
ground truth (`position_is_legal_slow`) by the `legalcheck` gate.

### Evaluation

A single seam — `evaluate(const Position *, EvalCache *)`, centipawns from the side-to-move's
perspective — is the one call site the network replaces.

- **NNUE (primary):** a king-bucketed **768×8 → 512** SCReLU perspective network. The perspective's own
  king square selects one of 8 input buckets (4 file-pairs × 2 board-halves), offsetting its 768-feature
  block. The accumulator is maintained **incrementally** inside `Position` and carries its net binding
  (`position_init(pos, net)`; the net itself is heap-loaded and owned by the `Engine`). A king move that
  changes a side's bucket triggers a refresh accelerated by a per-thread **finny cache** (owned by each
  `Searcher`, bound into its root position: a per-(perspective, bucket) cached accumulator plus the board
  it was built from, so a rebuild applies only piece diffs). The
  integer forward pass is AVX2-vectorised, and a shared lockless **eval cache** memoises results (its
  biggest win is qsearch stand-pat). The `nnuecheck` gate proves the incremental accumulator is
  bit-identical to a full refresh.
- **HCE (fallback):** a PeSTO tapered material + piece-square evaluation with bishop pair, mobility, and
  tempo terms, used when no net is loaded.

### Search

Fail-soft **principal-variation search** inside iterative deepening with **aspiration windows**:

- **Transposition table** — lockless for Lazy SMP: 16-byte `{key ^ data, data}` slots with an XOR
  torn-read guard, relaxed atomics, and a bit-field payload; depth-preferred replacement with generation
  aging. Mate scores cross the boundary as distance-from-node.
- **Move ordering** — TT move → captures (MVV-LVA) → promotions → killers → countermove → butterfly +
  continuation history; clearly losing captures are pruned by SEE rather than ordered late.
- **Quiescence** — captures and promotions only, SEE-pruned, with stand-pat.
- **Pruning & reductions** — null-move (adaptive reduction), reverse futility, futility, late-move
  pruning, late-move reductions (log-based table), SEE pruning, internal iterative reductions, and a
  pawn-keyed **eval correction history**.
- **Extensions** — check extensions and singular extensions (a TT-move exclusion search).
- Mate-distance pruning; repetition, fifty-move, and insufficient-material draw detection.
- **Lazy SMP** — independent whole-tree searches share the lockless TT and eval cache; everything else
  (history tables, killers, PV) is per-thread. `Threads` scales to 256.

All the pruning/reduction margins live in `SearchParams`, exposed as UCI spin options and **SPSA-tuned**
(`tools/spsa.py`); the tuned values are the shipped defaults.

### UCI & time management

The protocol loop (`uci.c`) runs the search on a coordinator thread that spawns the Lazy-SMP helpers;
`stop` sets a shared atomic. Options: `Hash`, `Clear Hash`, `Threads`, `Move Overhead`, `EvalFile`,
`OwnBook`/`BookFile`, plus the tunable search parameters. The same binary exposes the CLI self-tests
(`bench`, `perft`, `legalcheck`, `seecheck`, `fuzzcheck`, `bookcheck`, `nnuecheck`) and the training
helpers (`datagen`, `bullet2text`, `nnueeval`).

## NNUE training pipeline

The network is trained by an independent pipeline: `datagen` (or a public dataset) → `trainer/` (PyTorch,
CUDA) → a quantised `.nnue` → the engine loader. The single source of truth for the feature indexing and
quantisation contract is `trainer/features.py`, which `src/nnue.c` must reproduce byte-for-byte;
`trainer/verify.py` enforces a **0 cp** difference between the engine's integer eval and the trainer
reference. See [NNUE_TRAINING.md](NNUE_TRAINING.md) for the full contract and workflow.

```bash
# convert public bulletformat data to text, featurise to shard caches, then stream-train
./build/zenith bullet2text <shard.data> data/plenty/shard.txt 0 4
PYTHONPATH=trainer python trainer/train.py --featurise-shard data/plenty/shard.txt data/shards/s00.npz
PYTHONPATH=trainer python trainer/train.py --shard-dir data/shards --out nets/zenith.nnue \
    --hidden-size 512 --epochs 8 --batch-size 32768 --lr 1.2e-3 --wdl-lambda 0.3
PYTHONPATH=trainer python trainer/verify.py --net nets/zenith.nnue --fens <fens> --engine ./build/zenith
```

The streaming trainer (`--shard-dir`, one shard in RAM at a time) is what allows training on datasets far
larger than memory. `datagen` can also generate the engine's own self-play data
(`fen;stm_score;wdl` records) across all cores via `tools/datagen_parallel.sh`.

## Opening book

Standard **Polyglot** `.bin` books are supported (`src/book.c`; the Polyglot Zobrist key computation is
validated against the official spec vectors by `./build/zenith bookcheck`). No book ships in the repo — supply
your own, or fetch a free one with `make get-book` (downloads `performance.bin` from the GPL-3.0
python-chess repo into the gitignored `books/`):

```
make get-book                              # -> books/performance.bin (optional convenience)
setoption name BookFile value books/performance.bin
setoption name OwnBook value true          # default false — testing always runs bookless
```

## Provenance & attribution

Zenith is written from scratch and shares no source code with any other engine. It uses the same
published, textbook techniques every strong engine does — bitboards, magic bitboards, PVS, transposition
tables, null-move/LMR/futility pruning, SEE, NNUE, Lazy SMP (see the
[Chess Programming Wiki](https://www.chessprogramming.org)) — implemented independently.

- **NNUE network:** Zenith's own architecture (king-bucketed 768×8→512 SCReLU), its own trainer
  (`trainer/`), its own quantised file format (`ZNNUE3`), and its own trained weights. The shipped net was
  trained on the **public PlentyChess dataset**
  ([Yoshie2000/plentychess_data_bulletformat](https://huggingface.co/datasets/Yoshie2000/plentychess_data_bulletformat)) —
  a public dataset used to train Zenith's own network; no other engine's code or network is included or
  derived.
- **HCE fallback:** PeSTO piece-square tables (public, widely used).
- **Opening book:** the Polyglot key constants come from the public
  [book-format specification](http://hgm.nubati.net/book_format.html); no book is bundled.

## License

Zenith is free software licensed under the **GNU General Public License v3.0 or later** (see
[LICENSE](LICENSE)). Copyright © 2026 Jonny Reckless. You may use, modify, and redistribute it under the
terms of the GPL; derivative works must remain open under the same license.
