# Zenith

A from-scratch UCI chess engine in C++20. Independent design (shares no code or net format with the
sibling pawnstar engines). The classical core is complete and correct; NNUE + parallel search are the
next phases — see [DESIGN.md](DESIGN.md) for the full architecture and roadmap, and
[NNUE_TRAINING.md](NNUE_TRAINING.md) for the GPU training plan (the path to world-class strength).

## Build

Requires a C++20 compiler (clang++ 18 or g++ 13+). Magic bitboards keep the code portable; `-march=native`
is for local dev.

```bash
make            # -> ./zenith  (clang++, -O3 -flto -march=native)
make debug      # -> ./zenith-debug  (ASan + UBSan, for correctness work)
./zenith        # interactive UCI
```

## Verify

```bash
./zenith perft      # move-gen correctness vs known counts (startpos, Kiwipete, CPW 3/4/5)
./zenith bench 12   # fixed-depth node count + nps
make baseline       # snapshot ./zenith -> ./zenith-base for SPRT
tools/sprt.sh ./zenith ./zenith-base   # self-play SPRT of a change vs the baseline
```

`tools/sprt.sh` runs a self-play/cross-engine SPRT via **fastchess** (falls back to
`~/pawnstar_nnue/fastchess/fastchess`) with an openings book (`~/pawnstar_nnue/openings.epd` by default;
any EPD works — e.g. UHO from official-stockfish/books).

## NNUE (independent training pipeline)

Zenith trains its **own** NNUE from scratch — its own self-play data, its own PyTorch trainer, its own
`.nnue` format (no shared code, net, or data with any other engine). Everything plugs into the single
`evaluate()` seam behind the UCI `EvalFile` option.

```bash
# 1. Generate Zenith's own self-play data across all cores (fen;stm_score;wdl records)
tools/datagen_parallel.sh 7000 data/run 32 6000        # gamesPerWorker workers... nodes

# 2. Train a quantised net (PyTorch, CUDA); features/quantisation live in trainer/features.py
. .venv/bin/activate
PYTHONPATH=trainer python trainer/train.py --data 'data/run/shard_*.txt' \
    --out nets/zenith.nnue --cache data/run.npz --epochs 60 --wdl-lambda 0.4

# 3. Verify the engine's integer eval is bit-identical to the trainer (must be 0 cp)
PYTHONPATH=trainer python trainer/verify.py --net nets/zenith.nnue --fens data/run/shard_01.txt

# 4. Use it / test it
./zenith                                                # setoption name EvalFile value nets/zenith.nnue
CAND=./zenith CAND_NET=nets/zenith.nnue BASE=./zenith tools/sprt.sh   # NNUE vs HCE
```

The architecture is a 768→512 SCReLU perspective net (v1); the C++ side recomputes the accumulator each
eval (full refresh) — an incremental accumulator is the planned speed optimisation. See
[NNUE_TRAINING.md](NNUE_TRAINING.md) for the full contract and roadmap.

## Status (current)

- **Board/movegen:** magic bitboards (runtime-generated), copy-make, incremental Zobrist.
  **perft matches known counts** on startpos, Kiwipete, and CPW positions 3/4/5/6 (433M nodes, exact).
- **Search:** iterative deepening + aspiration, fail-soft PVS, transposition table (mate-adjusted bounds),
  quiescence + SEE, ordering (TT / MVV-LVA / killers / **continuation & butterfly history** / **countermove**),
  null-move, reverse futility, **futility**, late-move pruning, LMR, check extensions, mate-distance pruning,
  repetition / 50-move / insufficient-material draws.
- **Eval:** PeSTO tapered material+PST (+ bishop pair, mobility, tempo) behind a single `evaluate()` seam.
  **NNUE now plugs in here** (768→512 SCReLU perspective net) when a net is loaded via `EvalFile`, else HCE.
- **NNUE:** full independent pipeline — `datagen` self-play, PyTorch trainer, quantised loader + integer
  forward (verified bit-identical to the trainer). Nets improve with data/training; see NNUE_TRAINING.md.
- **UCI:** `uci` / `isready` / `ucinewgame` / `position` / `go` (clocks, movetime, depth, nodes, infinite,
  perft) / `setoption` (Hash, Clear Hash, Move Overhead, EvalFile) / `stop` / `quit`; `bench` + `perft` +
  `datagen` + `nnueeval` CLI.
- **Verified:** correct mate detection, sensible opening + endgame play, full self-play games to decisive
  results, ASan/UBSan-clean search. Single-threaded; ~1.8 Mnps (copy-make).

Strength: a solid classical engine (~2400–2700 class) with the full NNUE training pipeline now in place
and verified end-to-end. Closing the world-class gap is now a data + training program (bigger/cleaner
self-play data, less eval noise, larger nets, incremental accumulator), not new engine code — documented
in the two design files.

## Layout

```
src/types.h        colours, pieces, squares, packed Move, bit tricks
src/bitboard.*     attack tables + rook/bishop magic bitboards
src/position.*     board + mailbox, Zobrist, FEN, copy-make, attacks-to / in-check
src/movegen.*      pseudo-legal generation + legality filter (perft-gated)
src/eval.*         evaluate(): NNUE when a net is loaded, else tapered HCE
src/nnue.*         quantised NNUE loader + SCReLU integer forward (full-refresh accumulator)
src/datagen.*      self-play data generation (fen;stm_score;wdl) for NNUE training
src/tt.*           transposition table (depth-preferred, bounds)
src/search.*       ID + PVS + qsearch + ordering + pruning/reductions + time management
src/uci.*          protocol + bench + perft + datagen + nnueeval
src/main.cpp       entry
trainer/           independent PyTorch NNUE trainer (features.py, train.py, verify.py)
tools/sprt.sh              self-play / cross-engine SPRT harness (fastchess)
tools/datagen_parallel.sh fan datagen across CPU cores
tools/spsa.py             SPSA tuner for the search parameters (UCI-exposed) 
tools/link-memory.sh      wire Claude Code memory to the repo (run once per clone)
```

Project notes and lessons live in `.claude/memory/` (versioned in git). After cloning, run
`tools/link-memory.sh` once to point Claude Code's memory at the repo copy so those notes are shared.
