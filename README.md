# Zenith

A from-scratch UCI chess engine in **C17** with its own independently-trained NNUE evaluation.
Independent design throughout — no shared code, network, or data format with the sibling pawnstar
engines, which serve only as the reference opponent. **Zenith beats its reference engine at every
tested configuration** (single-thread and 8-thread, bullet and classical time controls).

Engine history: originally C++20, ported to C17 and verified bit-identical (the final C++ tree is
preserved at git tag `cpp-final`). See [DESIGN.md](DESIGN.md) for the architecture and
[NNUE_TRAINING.md](NNUE_TRAINING.md) for the training pipeline.

## Build

Requires a C17 compiler (clang 18 recommended; gcc 13+ works). Threads use C11 `threads.h` with a
pthread fallback (`src/platform.h`) for platforms without it (e.g. macOS). Magic bitboards keep the
code portable; `-march=native` is for local dev — a release build would fan out per microarch.

```bash
make            # -> ./zenith  (clang -std=c17, -O3 -flto -march=native)
make debug      # -> ./zenith-debug  (ASan + UBSan, for correctness work)
./zenith        # interactive UCI (prints "Zenith <major>.<minor>.<git commit count>")
```

## Verify

Every correctness property has an executable gate (all of these also run in CI):

```bash
./zenith perft       # movegen vs known counts: startpos/Kiwipete/CPW + the 128-position Ethereal suite
./zenith bench 13    # deterministic fixed-depth node signature (guards search behaviour) + nps
./zenith legalcheck  # fast legality/check predicates == copy-make ground truth over a perft walk
./zenith nnuecheck <net.nnue>   # incremental accumulator == full refresh, bit-identical
./zenith bookcheck   # Polyglot key computation vs the 9 official spec test vectors
make baseline        # snapshot ./zenith -> ./zenith-base for SPRT
tools/sprt.sh ./zenith ./zenith-base   # self-play SPRT of a change vs the baseline
```

`tools/sprt.sh` runs SPRTs via **fastchess** with an openings EPD. **No strength change lands without
passing an SPRT** — that discipline is the core of the project.

## NNUE (independent training pipeline)

Zenith's evaluation is a **king-bucketed 768×8 → 512 SCReLU perspective network** (format `ZNNUE3`):
the side's own king square selects one of 8 input buckets (4 file-pairs × 2 board-halves). The engine
maintains the accumulator **incrementally** inside `Position`, with a per-thread "finny" refresh cache
for king-bucket changes, and an AVX2 integer forward pass plus a shared eval cache.

The shipped net `nets/zenith-kb3.nnue` was trained on **1.4B positions** from the public PlentyChess
dataset with the repo's own PyTorch trainer. The full training loop is:

```bash
# Option A: generate own self-play data (fen;stm_score;wdl records) across all cores
tools/datagen_parallel.sh 7000 data/run 32 6000

# Option B (how kb3 was trained): convert public bulletformat data, then featurise + stream-train
./zenith bullet2text <shard.data> data/plenty/shard.txt 0 4
PYTHONPATH=trainer python trainer/train.py --featurise-shard data/plenty/shard.txt data/shards/s00.npz
PYTHONPATH=trainer python trainer/train.py --shard-dir data/shards --out nets/zenith.nnue \
    --hidden-size 512 --epochs 8 --batch-size 32768 --lr 1.2e-3 --wdl-lambda 0.3

# Verification gate (never skip): engine integer eval must equal the trainer bit-for-bit
PYTHONPATH=trainer python trainer/verify.py --net nets/zenith.nnue --fens <fens> --engine ./zenith
```

The contract lives in `trainer/features.py` (feature indexing + quantisation: QA=255, QB=64,
scale=400); `src/nnue.c` must reproduce it byte-for-byte, and `verify.py` enforces **0 cp** difference.
The streaming trainer (`--shard-dir`, one ~95M-position shard in RAM at a time) is what allows
training beyond the machine's RAM.

## Opening book

Standard **Polyglot** `.bin` books are supported (`src/book.c`; keys validated against the official
spec vectors via `./zenith bookcheck`). No book ships in the repo — supply your own:

```
setoption name BookFile value path/to/your-book.bin
setoption name OwnBook value true          # default false — testing always runs bookless
```

## Status

- **Board/movegen:** magic bitboards (runtime-generated), copy-make, incremental Zobrist + pawn key.
  Perft-exact on the full 133-position suite. Search filters pseudo-legal moves with a copy-free
  legality oracle (checkers/pins), differentially validated against copy-make ground truth.
- **Search:** iterative deepening + aspiration, fail-soft PVS, lockless XOR-guarded transposition
  table (bit-field payload), quiescence + SEE, ordering (TT / MVV-LVA / killers / countermove /
  butterfly + continuation history), null-move, reverse futility, futility, LMP, LMR, SEE pruning,
  check + singular extensions, IIR, mate-distance pruning, pawn-keyed eval correction history.
  Search parameters are UCI-exposed and **SPSA-tuned** (`tools/spsa.py`).
- **Parallel:** Lazy SMP (shared lockless TT + eval cache, per-thread history), `Threads` up to 256.
- **Eval:** king-bucketed NNUE (above) when a net is loaded via `EvalFile`, else a PeSTO tapered HCE.
- **UCI:** `uci`/`isready`/`ucinewgame`/`position`/`go` (clocks, movetime, depth, nodes, infinite,
  perft) / `setoption` (Hash, Clear Hash, Threads, Move Overhead, EvalFile, OwnBook, BookFile, plus
  the tunable search parameters) / `stop` / `quit`; versioned `id name Zenith <version>`.
- **Strength vs the reference engine (pawnstar, properly controlled matches):** **+78 Elo**
  single-thread and **+107 Elo** at 8 threads (tc 8+0.08); single-thread nps parity with a deeper
  effective search. ~2.4 Mnps single-thread with NNUE on a 13900HX.

## Layout

```
src/types.h        colors, pieces, squares, packed Move (uint16 + inline accessors), bit tricks
src/platform.h     C11 threads.h / pthread shim + monotonic clock
src/bitboard.*     attack tables + rook/bishop magic bitboards, BetweenBB/LineBB
src/position.*     bitboards + mailbox, Zobrist + pawn key, FEN, copy-make, legality/check oracles,
                   embedded NNUE accumulator (incrementally maintained)
src/movegen.*      pseudo-legal generation (+ legal wrapper for perft/datagen/UCI)
src/eval.*         evaluate(): NNUE when a net is loaded, else tapered HCE; shared eval cache
src/nnue.*         quantised king-bucketed NNUE loader + AVX2 integer forward + finny refresh cache
src/book.*         Polyglot opening book (key computation, probing, weighted move choice)
src/tt.*           lockless transposition table ({key^data, data} slots, bit-field payload)
src/search.*       ID + aspiration + PVS + qsearch + the full pruning/reduction/extension stack,
                   SPSA-tunable parameters, Lazy SMP entry points, time management
src/datagen.*      self-play data generation + bulletformat-to-text converter
src/uci.*          protocol loop + bench/perft/legalcheck/bookcheck CLI + version banner
src/version.h      major.minor + build number (git commit count, stamped by the Makefile)
src/main.c         entry
trainer/           independent PyTorch NNUE trainer (features.py, train.py, verify.py)
books/             Polyglot opening books (komodo.bin)
nets/              released .nnue networks (kb2, kb3)
tools/sprt.sh              SPRT harness (fastchess)
tools/datagen_parallel.sh  fan datagen across CPU cores
tools/spsa.py              SPSA tuner for the UCI-exposed search parameters
tools/link-memory.sh       wire Claude Code memory to the repo (run once per clone)
```

Project notes and lessons live in `.claude/memory/` (versioned in git). After cloning, run
`tools/link-memory.sh` once to point Claude Code's memory at the repo copy so those notes are shared.

## Provenance & attribution

Zenith is written from scratch and shares no source code with any other engine. It uses the same
published, textbook techniques every strong engine does (bitboards, magic bitboards, PVS, transposition
tables, null-move/LMR/futility pruning, SEE, NNUE, Lazy SMP) — see the
[Chess Programming Wiki](https://www.chessprogramming.org) — implemented independently.

- **NNUE network:** Zenith's own architecture (king-bucketed 768×8→512 SCReLU), its own trainer
  (`trainer/`), its own quantised file format (`ZNNUE3`), and its own trained weights. The shipped net
  was trained on the **public PlentyChess dataset**
  ([Yoshie2000/plentychess_data_bulletformat](https://huggingface.co/datasets/Yoshie2000/plentychess_data_bulletformat)) —
  a third-party open-source engine's public self-play data, used to train Zenith's own network; no other
  engine's code or network is included or derived.
- **HCE fallback:** PeSTO piece-square tables (public, widely used).
- **Opening book:** the Polyglot key constants come from the public
  [book-format specification](http://hgm.nubati.net/book_format.html); no book is bundled.

If you find any concrete originality concern, please open an issue — transparency is the intent.

## License

Zenith is free software licensed under the **GNU General Public License v3.0 or later** (see
[LICENSE](LICENSE)). Copyright © 2026 Jonny Reckless. You may use, modify, and redistribute it under the
terms of the GPL; derivative works must remain open under the same license.
