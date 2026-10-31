# Zenith

An independent UCI chess engine in **C17** — its own search, move generation and data generation,
written from scratch rather than ported — evaluating with the shared Reckless NNUE network.

The source code was written by Claude Code under the close supervision and direction of Jonny Reckless.

Zenith pairs a modern alpha-beta search — iterative deepening, principal-variation search, a lockless
transposition table, the full pruning/reduction stack, and Lazy SMP — with a king-bucketed neural
network evaluation: the **shared Reckless NNUE**, which every chess engine in the organisation now
evaluates with, trained on Leela Chess Zero self-play data. The network shape is Zenith's own design,
measured here; the weights it ships come from the shared trainer in
[`nnue-training`](https://github.com/reckless-technology/nnue-training), and the nets it shipped before
3.0.0 were trained on PlentyChess positions — see
[Provenance of earlier networks](NNUE_TRAINING.md#provenance-of-earlier-networks).

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

**Setting up a machine.** `tools/setup-dev.ps1` (Windows) and `tools/setup-dev.sh` (Linux, macOS) install what
building, testing, debugging and formatting need - on Windows MSYS2 at `C:\msys64` with its CLANG64 clang, lld,
compiler-rt, llvm-tools, python and gdb, and make, git and zip; on Linux apt's clang, make, python3, git, zip and gdb;
on macOS Xcode's command-line tools - plus clang-format 22.1.8 through uv, the git hooks and the recommended VS Code
extensions, checking each first, so they are safe to run again; `-Check` / `--check` only reports. In VS Code they
are the `setup environment` and `check environment` tasks. There is no training environment to install here any
more — the trainers live in `nnue-training`, which carries its own `pytorch/requirements.txt`.

```bash
make                 # -> ./build/zenith        (release: clang -std=c17, -O3 -flto -march=native, whole-program LTO)
make ARCH=x86-64-v2  # portable release build: override the default -march=native (see .github/workflows/release.yml)
make debug           # -> ./build/zenith-debug  (AddressSanitizer + UBSan, -O1 -g; use for movegen/make_move work)
make pext            # -> ./build/zenith-pext   (BMI2 PEXT sliding attacks; bit-identical, ~2% faster on Haswell+/Zen3+)
make datagen         # -> ./build/zenith-datagen + ./build/zenith-bullet2text (standalone NNUE training-data tools)
make datagen-debug   # -> ASan+UBSan builds of both datagen tools (they live outside src/, so `make debug` skips them)
make check           # build + run every correctness gate (mirrors CI; see Verify below)
make coverage        # line coverage of src/ by those gates, failing under 90 % (LLVM source-based; see Verify below)
make perft           # build + run the perft movegen gate
make bench           # build + run the fixed-depth node-signature benchmark
make baseline        # snapshot the current ./build/zenith -> ./build/zenith-base (the SPRT reference binary)
make get-book        # download a free Polyglot opening book -> books/ (gitignored); prints the setoption lines
make tables          # regenerate the committed constant tables (src/generated/*.inc: Zobrist, Q28 ln,
                     #   bitboard geometry, magic multipliers) — a
                     #   deliberate step, never a build side effect; the bench signature guards the values
make format          # clang-format all sources in place (src/ and datagen/) — use clang-format 22.1.8
make hooks           # install the clang-format pre-commit hook (once per clone; core.hooksPath -> .githooks)
make doc             # -> doc/html/index.html (Doxygen API reference)
make clean           # remove build outputs (binaries + doc/html)
./build/zenith       # interactive UCI (prints "Zenith <major>.<minor>.<build>"; see Versions and releases)
```

**Windows** builds natively in [MSYS2](https://www.msys2.org/)'s CLANG64 environment, as CI's Windows job does:
install MSYS2 at `C:\msys64`, then `pacman -S --needed mingw-w64-clang-x86_64-clang mingw-w64-clang-x86_64-lld
mingw-w64-clang-x86_64-python make git` (and `mingw-w64-clang-x86_64-gdb` to debug), and every target above runs as written
from a CLANG64 shell, `make debug`'s AddressSanitizer and UBSan included. The binaries are `build/zenith.exe` and
friends; they load `libwinpthread-1.dll` (and the debug build the ASan runtime) from `C:\msys64\clang64\bin`, which a
CLANG64 shell has on its PATH. MSVC is not supported.

**VS Code** works on all three platforms: `.vscode/tasks.json` runs these make targets (on Windows in MSYS2's CLANG64
bash), `.vscode/launch.json` has the debug, release and attach launches per platform (gdb on Linux and Windows, lldb
through CodeLLDB on macOS), and `.vscode/c_cpp_properties.json` gives IntelliSense each platform's compiler.

## Verify

Every correctness property has an executable gate. `make check` runs them all; CI runs the same gates on
Linux and Windows on every push (asserting the bench signature on each), plus a threaded UCI smoke and the
NNUE gates against a synthetic net, and holds their line coverage of `src/` to 90 % on Linux (below); macOS runs
the full suite on release candidates. Any failure aborts non-zero.

```bash
make check                          # build + run every gate below
./build/zenith perft                # movegen vs known counts: canonical + ep/castling/promotion catchers + Ethereal 128
./build/zenith bench 13             # deterministic fixed-depth node signature (guards search behaviour) + nps
./build/zenith legalcheck           # the copy-free legality/check predicates == copy-make ground truth over a perft walk
./build/zenith seecheck             # static exchange evaluation vs hand-verified capture positions
./build/zenith fuzzcheck            # malformed-FEN/UCI hardening (memory safety; run the `make debug` build under ASan)
./build/zenith nnuecheck <net.nnue> # incremental accumulator == full refresh, bit-identical
./build/zenith bookcheck            # Polyglot key computation vs the 9 official spec test vectors
./build/zenith license              # GPL notice + the ODbL 1.0 attribution the embedded net's training data owes
./build/zenith embookcheck          # the build-time-embedded book: blob, entry invariants, probe path
python3 tools/uci_check.py ./build/zenith  # the UCI front end through pipes (below)
```

`tools/uci_check.py` tests what a GUI, the book builder or a test harness talks to, by running the binary as they
do: every UCI option (each shown to reach the search where it should, by the node count of a fixed-depth search -
the same net from `EvalFile` searches exactly as the embedded one), the `position` and `go` forms (a clock, a fallen
flag, nodes, movetime, perft, infinite and stop, a mate, a mated position, a history past its cap), crafted Polyglot
books (castling entries, and the books the loader must refuse), the notation subcommands against hand-verified
answers (`moves`, `polykey` against the specification's vectors, every SAN form, `applymoves`), `nnueeval` with the
net's colour symmetry, every verdict the `RNNUE1` loader can reach (a bad magic, a retired container, a corrupt
header, a sound header describing another architecture, a non-zero reserved byte, a truncated, over-long or empty file,
a corrupt payload), and
malformed input. It needs only Python 3, which the build already does.

`main` is protected: changes land via pull request with all CI checks green.

**Coverage.** `make coverage` measures how much of `src/` those gates reach, with LLVM's source-based coverage, and
fails under `COVERAGE_MIN` per cent of its lines (90 by default; CI's `coverage` job runs it on Linux). It builds an
instrumented engine (`-fprofile-instr-generate -fcoverage-mapping`, `-O2 -g`, the same sources and embedded net and
book) into `build/cov/`, runs every `make check` gate against it - nnuecheck included, and the front-end gate with
`--terminal`, which on Linux and macOS also runs legalcheck on a pseudo-terminal to test its in-place progress line -
plus CI's threaded UCI smoke, every process merging its counts into one profile under a file lock
(`LLVM_PROFILE_FILE=build/cov/zenith-%m.profraw`: a file per pid lost profiles on Windows, which reuses pids within a
run), then prints `llvm-cov report` over `src/`, leaving `build/cov/lcov.info` and `build/cov/html/index.html` beside
it. The figure is `src/` alone - the engine every release ships, its UCI front end and its headers; the developer
tools no release ships, `datagen/` (zenith-datagen, bullet2text) and `tools/`, are not measured. The bench
gate still holds its node signature: instrumentation counts and changes nothing the search computes. It needs clang
with its profile runtime and `llvm-profdata` / `llvm-cov`: Xcode's through `xcrun` on macOS, MSYS2 CLANG64's
`llvm-tools` on Windows (`tools/setup-dev.ps1` installs them), and on Linux `llvm-profdata` on the PATH or, failing
that, the copy versioned like clang (`llvm-profdata-18` beside clang 18; `LLVM_SUFFIX=-18`, or `LLVM_PROFDATA=` and
`LLVM_COV=`, name them). The gates reach 94.6 % of `src/`'s lines on Linux, 94.3 % on macOS and 94.0 % on
Windows; each platform compiles its own branches of `platform.h`, and a tenth of a point moves with timing (a search
stopped inside quiescence or not). What they leave is code no gate can reach without breaking the engine:
allocation failures (the engine, the net, the book, the search pool, a thread that will not start), the failure
branches of the gates themselves (a mismatch report), a `Hash` above the 65536 MB cap (whose test would ask a shared
runner for 64 GB), the Windows console's width on Linux and macOS (and the terminal itself on Windows), and a few
helpers nothing calls.

The **`bench` node signature** is the linchpin: a fixed-depth search produces a deterministic node count,
so any unintended change to search behaviour shows up immediately, and a pure-speed change is proven safe
by leaving it identical. `tools/sprt.sh` runs a self-play SPRT (via **fastchess**) to decide whether a
change that *does* alter the signature is actually a strength gain — no strength change lands on intuition.

## Versions and releases

The version is `MAJOR.MINOR.BUILD`, from git alone, the rule all of reckless-technology's projects follow:

- `MAJOR.MINOR` is the nearest annotated tag `vMAJOR.MINOR` reachable from the commit built (a *version mark*, made by
  hand on `main`);
- `BUILD` is the number of commits since that tag, the tagged commit itself being 0.

`tools/version.sh` works it out (`git describe --tags --match 'v*.*' --exclude 'v*.*.*'`). The `Makefile` compiles it in
as `ZENITH_VERSION_STRING`, and rebuilds when it changes. The engine reports it: `id name Zenith 2.0.0`, and in the
banner. Nothing in the tree holds a version to edit, so no version-bump commit follows a release. A tree with no mark
reachable (a source tarball, a shallow clone, a fork without the tags) builds as `0.0.0`.

`.github/workflows/release.yml` makes a release:

- **Pushing a mark** (`git tag -a v2.1 -m "Zenith 2.1" && git push origin v2.1`) releases `2.1.0`.
- **Running it by hand** on `main` (`gh workflow run release.yml --ref main`) releases `main`'s head as
  `MAJOR.MINOR.BUILD`. `-f dry_run=true`, from any branch, builds, tests and signs the macOS engine but publishes
  nothing.

It refuses a mark that isn't `vMAJOR.MINOR`, a commit not on `main`, and a version already released. Each platform is
built from nothing on a GitHub-hosted runner (`ubuntu-24.04`, `windows-2025`, `macos-15`), as is CI's every job.
`make check` runs on the portable build of each, and the bench signature on the PEXT build, and each engine shipped
must report the version. The workflow creates the release
`vMAJOR.MINOR.BUILD` and its tag itself. It never runs for a pull request.

| File | What it is |
|---|---|
| `zenith-v<version>-windows-x86-64-v2.zip`, `-windows-x86-64-v3-pext.zip` | `zenith.exe` Authenticode-signed through SSL.com eSigner (Jonathan Reckless) |
| `zenith-v<version>-macos-arm64.zip` | the engine for Apple silicon, signed with a Developer ID (hardened runtime) and notarised by Apple |
| `zenith-v<version>-linux-x86-64-v2.tar.gz`, `-linux-x86-64-v3-pext.tar.gz` | the engine for Linux |
| `zenith-v<version>-SHA256SUMS.txt` | the checksums of every archive |

Each archive holds the engine, `README.md`, `LICENSE` and `THIRD-PARTY-NOTICES.md`. `x86-64-v2` runs on
essentially any modern x86-64;
`x86-64-v3-pext` needs AVX2 and BMI2 (Intel Haswell+, AMD Zen 3+) and is the one to run where it can: the NNUE's
SIMD is AVX2, so the `x86-64-v2` engine evaluates in scalar code and searches about a third slower (bench 1.30 against
1.95 Mnps on one P-core of an i9-13900HX). A bare macOS binary can't carry a stapled
notarisation ticket, so Gatekeeper looks its ticket up online. The signing runs from `tools/sign-macos.sh` and
`tools/sign-windows.ps1`, with the organisation's signing secrets (the same ones as its other apps). The Windows engines
are signed only in a run that publishes, since every signing spends the eSigner account's allowance. A publishing run
that would ship an unsigned engine fails.

## Architecture

One translation unit per file, flat `src/`. **No globals:** all mutable engine state lives in an `Engine`
aggregate (`engine.h` — the transposition table, the shared search state, the eval cache, and the loaded
NNUE net), built by `engine_new()` and released by `engine_delete()` in `main`, passed explicitly
everywhere; UCI session state is a `UciSession` on `uci_loop`'s stack. The Zobrist keys, Q28 ln table, and the leaper/geometry attack tables
are generated compile-time constants (`tools/generate_tables.py` → `src/generated/*.inc`). The one
exception is
`bitboard.c`'s ~850KB magic sliding-attack tables, deterministically filled from generated constant
multipliers by `init_bitboards()` — idempotent, called lazily by `engine_new()`, with no search or
PRNG — before any thread exists.

| File | Contains |
|---|---|
`src/types.h`      | Color, Piece (NO_PIECE=0..KING=6), Square, packed 16-bit Move + inline accessors, bit helpers
`src/platform.h`   | C11 threads.h / pthread thread shim + a monotonic clock
`src/bitboard.*`   | precomputed pawn/knight/king attacks, BetweenBB/LineBB, rook/bishop MAGIC bitboards
`src/position.*`   | bitboards + mailbox, Zobrist + pawn key, FEN I/O, copy-make, legality/check oracles
`src/movegen.*`    | one masked setwise generator: pseudo-legal + single-pass legal instantiations
`src/engine.*`     | the Engine aggregate (TT + shared search state + eval cache + NNUE net) + its new/delete
`src/eval.*`       | evaluate(): the NNUE forward pass (a net is always loaded); shared eval cache
`src/nnue.*`       | king-bucketed quantised NNUE: loader, feature indexing, AVX2 forward, finny refresh cache
`src/book.*`       | Polyglot opening book: key computation, probing, weighted move choice
`src/tt.*`         | lockless transposition table ({key^data, data} slots, bit-field payload)
`src/search.*`     | iterative deepening, PVS, quiescence, the pruning/reduction/extension stack, Lazy SMP
`src/uci.*`        | the UCI protocol loop, time manager, and CLI self-test subcommands
`src/version.h`    | the version string, MAJOR.MINOR.BUILD from the git tags (`tools/version.sh`, stamped by the Makefile)
`src/main.c`       | entry + CLI dispatch
`datagen/*`        | standalone training-data tools (`make datagen`): self-play generation + a bulletformat-to-text converter

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
`position_is_move_legal(move, checkers, pinned)`, built on precomputed pins and checkers — *before* it pays for
`make_move`, so illegal and pruned moves never cost a copy. `generate_legal` instantiates the same skeleton
with real check-evasion and pin-ray masks baked into the target sets (per-destination king safety; a full
test only for en passant), producing fully legal moves in a single pass with no filter — this backs perft
(~550 Mnps), datagen, and UCI move parsing. The oracles (`position_is_move_legal`, `pinned_to_king`,
`gives_check_fast`, `discovered_check_candidates`) are differentially validated against the copy-make
ground truth (`position_is_move_legal_slow`) by the `legalcheck` gate.

### Evaluation

A single seam — `evaluate(const Position *, EvalCache *)`, centipawns from the side-to-move's
perspective — is the one call site the network replaces.

- **NNUE (primary):** a king-bucketed **768×8 → 1024 → 8 output heads** SCReLU perspective network. A
  perspective whose own king sits on files e–h is mirrored horizontally (`square ^ 7`) onto files a–d; the
  (mirrored) king square selects one of 8 input buckets (4 files × 2 board-halves), offsetting its
  768-feature block, and the total piece count selects one of 8 material output heads. The accumulator is
  maintained **incrementally** inside `Position` and carries its net binding
  (`position_init(pos, net)`; the net itself is heap-loaded and owned by the `Engine`). A king move that
  changes a side's bucket or mirror state triggers a refresh accelerated by a per-thread **finny cache**
  (owned by each `Searcher`, bound into its root position: a per-(perspective, mirror, bucket) cached
  accumulator plus the board it was built from, so a rebuild applies only piece diffs). The
  integer forward pass is AVX2-vectorised, and a shared lockless **eval cache** memoises results (its
  biggest win is qsearch stand-pat). The `nnuecheck` gate proves the incremental accumulator is
  bit-identical to a full refresh.
- **Embedded by default:** the shipped net is generated into a C array at build time (`tools/embed_net.py`
  -> `build/embedded_net.o`) and linked into every binary, so a bare executable is full strength with no
  external files; `EvalFile` swaps in another net at runtime. A net that fails any of the container's checks
  is **refused with no fallback**: the net already loaded keeps playing, and the engine says so on *both*
  streams — the loader's verdict (the field, the file's value and the engine's) and which net is in play
  instead, as `info string` lines on stdout for a GUI and on stderr for a match harness. (The former PeSTO
  hand-crafted fallback was removed with the embedded default; it remains in git history.)

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
(`bench`, `perft`, `legalcheck`, `seecheck`, `fuzzcheck`, `bookcheck`, `embookcheck`, `nnuecheck`) and the NNUE
verification helper (`nnueeval`); the training-data tools are separate executables built by `make datagen`.
`license` is a command of the protocol as well as a subcommand (`licence` is accepted for either): it prints
the GPL notice and the ODbL 1.0 attribution the embedded network's training data obliges, so a GUI or a match
harness can ask the binary for it. It is the standard command across the Reckless engines, and it prints
only - no engine or search state is read or written.

## NNUE training pipeline

**The trainers live in [`nnue-training`](https://github.com/reckless-technology/nnue-training)** — the
single source for the Reckless NNUE: the architecture, the king-bucket map, the `RNNUE1` container, both
trainers and the data pipeline. Zenith is one of its consumers, and the split is *producer asserts,
consumer verifies* — that repository asserts what a net is, and this engine checks every header field
against its own compiled constants and refuses loudly, with no fallback, on a mismatch. What lives here
is the loader (`src/nnue.*`), the net tools in `tools/`, and the engine-side gates.

So the pipeline is `datagen` (or an external dataset via `bullet2text`) → the shared trainer → a quantised
`.nnue` in the `RNNUE1` container (`tools/stamp_net.py`) → the engine loader. The shipped net need not come
from Zenith's own data: any net of the shared shape loads, and **the net shipped from
3.0.0, `nets/reckless-lc0-runf-sb870.nnue`, is trained entirely on Leela Chess Zero self-play data**, from
the shared Reckless NNUE pipeline — data published under the **Open Database License 1.0**, whose
attribution notice is in [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) and printed by
`./build/zenith license`. The nets up to `zenith-cap2` were trained on **PlentyChess**
positions — see [Provenance of earlier networks](NNUE_TRAINING.md#provenance-of-earlier-networks).

The single source of truth for the feature indexing and quantisation contract is `nnue-training`'s
`pytorch/features.py`, which `src/nnue.c` must reproduce byte-for-byte; its `pytorch/verify.py` enforces a
**0 cp** difference between the engine's integer eval and the Python reference. One file is deliberately
copied into this repository — `tools/rnnue1.py`, verbatim `nnue-training/spec/rnnue1.py`, because three
tools here import it and must run on a CI runner with no network: change it in both or in neither. See
[NNUE_TRAINING.md](NNUE_TRAINING.md) for the full contract, and `nnue-training`'s README for how to train.

```bash
# the engine's side: data in, net verified and gated here
make datagen         # -> ./build/zenith-datagen + ./build/zenith-bullet2text
./build/zenith-bullet2text <shard.data> data/bullet/shard.txt 0 4   # bulletformat -> text
tools/datagen_parallel.sh 20000 data/run1 32 5000 8                 # or Zenith's own self-play
./build/zenith nnuecheck nets/zenith.nnue                           # incremental == full refresh
# the 0-cp contract gate, run from a nnue-training clone:
python3 pytorch/verify.py --net nets/zenith.nnue --fens <fens> --engine /path/to/zenith/build/zenith
```

`./build/zenith-datagen` generates the engine's own self-play data (`fen;stm_score;wdl` records) across all
cores via `tools/datagen_parallel.sh`; `bullet2text` converts a bulletformat dataset to the same text.

## Opening book

Standard **Polyglot** `.bin` books are supported (`src/book.c`; the Polyglot Zobrist key computation is
validated against the official spec vectors by `./build/zenith bookcheck`). The repo ships Zenith's own
book, `books/zenith-book-r5.bin` (487KB, ~14,000 positions) — built entirely from Zenith self-play by
`tools/build_book.py` (screen/rescore every move with the engine, self-play playouts for near-ties, negamax
backup over the position DAG), worth ~+30 Elo of clock savings and opening variety in self-play gates.
Beyond the moves Zenith plays, it carries **opponent coverage**: prepared replies to plausible moves
Zenith would not choose itself (1. d4, 1. c4, and ~14k others), so it stays in book against foreign
repertoires. r5 is the builder's r4 with one move added by hand (`tools/add_book_moves.py`): 1. d4 at the root,
weight 42 beside 1. e4's 115 and 1. c4's 54, so Zenith opens 1. d4 about a fifth of the time (42 / 211) and
plays on into the ~2,100 positions r4 already held after it. The builder's `zenith-book-r4.bin` and the earlier,
coverage-free `zenith-book-r3.bin` are retained alongside.
Inspect it with `tools/show_book.py` (one SAN line of play per row). The book is **embedded in every
binary** (like the net), so `setoption name OwnBook value true` alone enables it — no BookFile needed;
`BookFile` loads an external book instead. OwnBook stays **off by default**: testing is bookless.
Alternatively supply your own book, or fetch a free one with `make get-book` (downloads
`performance.bin` from the GPL-3.0
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

- **NNUE network:** the shape Zenith designed and measured (king-bucketed, horizontally-mirrored
  768×8→1024→8 SCReLU), now the **shared Reckless NNUE** that every engine in the organisation evaluates with,
  in the `RNNUE1` container they all read. **The network shipped from 3.0.0
  (`nets/reckless-lc0-runf-sb870.nnue`) is trained entirely on Leela Chess Zero self-play data**, and comes
  from the shared Reckless NNUE training pipeline (`nnue-training`) rather than from any trainer of
  Zenith's own — which is why it does not carry a `zenith-` name. No other engine's **code** is included or derived.

  **Training-data attribution (ODbL 1.0).** That data is published at
  <https://storage.lczero.org/files/training_data/> under the **Open Database License 1.0**, with its
  contents under the **Database Contents License 1.0**. A trained network is an ODbL *Produced Work*, and
  distributing one obliges a notice naming the source:

  > Position data from the Leela Chess Zero self-play training data
  > (https://storage.lczero.org/files/training_data/), run test91, used under the Open Database
  > License 1.0 (http://opendatacommons.org/licenses/odbl/1.0/).

  > Position data from the Leela Chess Zero self-play training data
  > (https://storage.lczero.org/files/training_data/), run test60, used under the Open Database
  > License 1.0 (http://opendatacommons.org/licenses/odbl/1.0/).

  The pool was 62.83 % `test91` and 37.17 % `test60`. The obligation is **the notice**: ODbL does *not*
  require the network to be ODbL-licensed, and the network is *not* GPL merely because GPLv3 code trained it
  or reads it. The full notice, and every other notice this engine owes, is in
  [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md); `./build/zenith license` prints it from the binary.

  **Provenance of earlier networks.** Zenith networks up to and including **zenith-cap2** were trained on
  positions from a public self-play dataset generated by **PlentyChess**, by Yoshie2000 — up to 2.15
  billion positions. That data is no longer used, and those nets are no longer in the tree, so no release
  ships one. The attribution is kept here deliberately: those networks were measurably stronger for having
  been trained on that data, and the record should show where they came from. The full lineage, with every
  measurement as it was made, is in
  [NNUE_TRAINING.md](NNUE_TRAINING.md#provenance-of-earlier-networks).
- **Opening book:** the Polyglot key constants come from the public
  [book-format specification](http://hgm.nubati.net/book_format.html); the book Zenith ships and embeds,
  `books/zenith-book-r5.bin`, is built entirely from Zenith's own self-play, and no third-party book is
  bundled.
- **Every notice in one place:** [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) — the ODbL training-data
  notice, the Ethereal perft reference counts, the Polyglot format specification, and the earlier networks'
  data. It ships in every release archive beside the engine and the licence.

## License

Zenith is free software licensed under the **GNU General Public License v3.0 or later** (see
[LICENSE](LICENSE)). Copyright © 2026 Jonny Reckless. You may use, modify, and redistribute it under the
terms of the GPL; derivative works must remain open under the same license.

Third-party notices are in [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md), which ships in every release
archive and is printed by `./build/zenith license`. Chief among them: the network embedded in every binary
was trained on Leela Chess Zero self-play data, published under the **Open Database License 1.0**, and
distributing the network obliges the attribution notice that file carries. The network is not itself
ODbL-licensed and is not GPL.
