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

`tools/sprt.sh` needs `cutechess-cli` on PATH and an openings book (`~/pawnstar_nnue/openings.epd` by
default; any EPD works — e.g. UHO from official-stockfish/books).

## Status (current)

- **Board/movegen:** magic bitboards (runtime-generated), copy-make, incremental Zobrist.
  **perft matches known counts** on startpos, Kiwipete, and CPW positions 3/4/5/6 (433M nodes, exact).
- **Search:** iterative deepening + aspiration, fail-soft PVS, transposition table (mate-adjusted bounds),
  quiescence + SEE, ordering (TT / MVV-LVA / killers / **continuation & butterfly history** / **countermove**),
  null-move, reverse futility, **futility**, late-move pruning, LMR, check extensions, mate-distance pruning,
  repetition / 50-move / insufficient-material draws.
- **Eval:** PeSTO tapered material+PST (+ bishop pair, mobility, tempo) behind a single `evaluate()` seam —
  **this is where NNUE plugs in**.
- **UCI:** `uci` / `isready` / `ucinewgame` / `position` / `go` (clocks, movetime, depth, nodes, infinite,
  perft) / `setoption` (Hash, Clear Hash, Move Overhead) / `stop` / `quit`; `bench` + `perft` CLI.
- **Verified:** correct mate detection, sensible opening + endgame play, full self-play games to decisive
  results, ASan/UBSan-clean search. Single-threaded; ~1.8 Mnps (copy-make).

Strength: a solid classical engine (~2400–2700 class). The world-class gap is NNUE + tuning + testing,
not code — that program is documented in the two design files.

## Layout

```
src/types.h        colours, pieces, squares, packed Move, bit tricks
src/bitboard.*     attack tables + rook/bishop magic bitboards
src/position.*     board + mailbox, Zobrist, FEN, copy-make, attacks-to / in-check
src/movegen.*      pseudo-legal generation + legality filter (perft-gated)
src/eval.*         evaluate(): tapered HCE now, NNUE later
src/tt.*           transposition table (depth-preferred, bounds)
src/search.*       ID + PVS + qsearch + ordering + pruning/reductions + time management
src/uci.*          protocol + bench + perft
src/main.cpp       entry
tools/sprt.sh      self-play SPRT harness
```
