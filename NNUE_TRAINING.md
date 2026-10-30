# Zenith NNUE — architecture & training guide

Zenith evaluates positions with the **shared Reckless NNUE**: one network shape and one file format
(`RNNUE1`) for every chess engine in the organisation, so a net trained once loads in all of them and a new
evaluation function is developed once rather than once per engine (Jonny, 2026-10-27). The shape is
Zenith's own — its two additions over the pawnstar engines' were measured and SPRT-gated here — and the
container is pawnstar's, extended with the bucket map's contents and an integrity check. Zenith keeps its
own trainer (`trainer/`, PyTorch) and its own data; the skill `reckless-nnue` is the specification. This
document is the reference for how the network is built and how to train a new one.

The pipeline is four stages:

```
   data                             trainer (PyTorch/CUDA)             engine (C)
 ┌──────────────────────┐  text     ┌──────────────────────┐  .nnue   ┌────────────────────┐
 │ ./build/             │ ────────▶ │ trainer/train.py     │ ───────▶ │ src/nnue.c loader  │
 │  zenith-datagen      │  fen;     │  featurise → SCReLU  │  RNNUE1  │  + integer forward │
 │  (self-play)         │  score;   │  net → quantise      │          │  behind evaluate() │
 │  OR                  │  wdl      │                      │          │                    │
 │ ./build/             │           └──────────────────────┘          └────────────────────┘
 │  zenith-bullet2text  │                      │                                │
 │  (public data)       │                      └────── trainer/verify.py ───────┘
 └──────────────────────┘                  engine int eval == trainer ref (0 cp)
```

`evaluate()` in `src/eval.c` is the single seam: it is always the NNUE forward pass. Every engine holds a
net — the shipped one is embedded into the binary at build time (`tools/embed_net.py`; `NET` in the
Makefile), and the UCI `EvalFile` option swaps in another at runtime. A net that fails any of the
container's checks is **refused**, loudly and with no fallback: the net that was playing keeps playing and
nothing is silently substituted for the net that was asked for. "Loudly" means on both streams — a refused
`setoption name EvalFile` prints the loader's reason *and* the name of the net still in play as `info string`
lines on stdout and as the same lines on stderr, because a GUI never sees stderr and a match harness
routinely drops `info` lines. No search change is needed to adopt or swap a net.

---

## 1. Network architecture

A **perspective network** with **king-input buckets** — the design most strong hobby engines converge on:

```
per-perspective inputs : 768  =  2 colors × 6 piece types × 64 squares   (own pieces first)
king mirroring         : if the perspective's OWN king is on files e-h, the whole perspective is
                         mirrored horizontally (every square ^ 7) — the net only ever sees a king
                         on files a-d
king buckets           : the (mirrored) king square selects 1 of 8 buckets
                         (4 files {a, b, c, d} × 2 board-halves {ranks 1-4, 5-8}),
                         offsetting the 768 block → 768 × 8 = 6144 feature-transformer rows
feature transformer    : 6144 → 1024    (one 1024-wide accumulator per perspective)
concatenate            : [own(1024), opponent(1024)] = 2048
activation             : SCReLU(x) = clamp(x, 0, 1)²
output buckets         : 1024 → 8 heads; the head is selected by total piece count
                         ((popcount(occupied) − 2) / 4), dequantised to centipawns
```

Two accumulators are maintained, one per perspective (side-to-move "own" and the opponent). Black's
perspective mirrors the board vertically (`square ^ 56`) and swaps the own/opponent colour halves, so the
network only ever learns "from the side to move's point of view." Horizontal king mirroring folds the
board's left/right symmetry the same way: instead of spending 4 of 8 buckets on kingside king placements,
mirrored positions share weights, so every bucket trains on twice the effective data at the same file size.

The feature-index convention lives in **`trainer/features.py`** (`feature_index`, `king_bucket`,
`king_mirror`) and is the single source of truth; `src/nnue.c` reproduces it exactly. The architecture's
numbers, the king-bucket map and the container itself live one level down in **`trainer/rnnue1.py`**, which
is stdlib-only so that every tool can import it; `features.py` takes its constants from there, and
`src/nnue.c`'s `KING_BUCKET_MAP` is a copy of the same 64 entries the container digests. Feature layout
within one king bucket:

```
index = king_bucket(rel_king) × 768  +  rel_color × 384  +  piece_type × 64  +  rel_square
        rel_color  = 0 for the perspective's own pieces, 1 for the opponent's
        rel_square = square (White perspective)  or  square ^ 56 (Black perspective),
                     then ^ 7 for every piece when king_mirror(rel_king) (own king on files e-h)
        piece_type = 0..5  (pawn..king, 0-based on the wire)
```

Hidden size is a knob in the trainer (`--hidden-size`), but the shared container fixes it at **1024**
along with everything else above: a net of any other width carries a header the engines refuse.

---

## 2. Quantisation & on-disk format (`RNNUE1`)

The trainer exports a fixed-point `int16` network; the engine loads it directly and does an all-integer
forward pass (a float `.pt` checkpoint is saved alongside for inspection only).

```
QA = 255   feature-transformer weight / accumulator scale
QB = 64    output-weight scale
EVALUATION_SCALE = 400   logit → centipawn scale (must equal the trainer loss scale)

FT weights : round(w × QA)      → int16   [6144][1024]  (feature-major, king-bucketed: 8 × 768 rows)
FT bias    : round(b × QA)      → int16   [1024]
OUT weights: round(w × QB)      → int16   [8][2048]      (bucket-major; per bucket: own half [0,1024)
                                                          then opponent half [1024,2048))
OUT bias   : round(b × QA × QB) → int32   [8]
```

**File `nets/<name>.nnue`** (little-endian): a **64-byte `RNNUE1` header**, then those four arrays
back-to-back as the 12,617,760-byte payload, with nothing after it. The header describes the net rather
than leaving it to be guessed from the file's length:

| Offset | Size | Field | At this shape |
|---|---|---|---|
| 0 | 8 | `magic` | `"RNNUE1\0\0"` |
| 8 | 2 | `format_version` | 2 (the mirrored map with material output heads; 1 is the unmirrored one) |
| 10 | 2 | `header_bytes` | 64 — lets a later version extend the header without breaking this reader |
| 12 | 2 | `features_per_bucket` | 768 |
| 14 | 2 | `king_buckets` | 8 |
| 16 | 2 | `hidden_size` | 1024 |
| 18 | 2 | `output_buckets` | 8 |
| 20 | 1 | `flags` | bit 0 = horizontal king mirroring, set; every other bit zero |
| 21 | 1 | `activation` | 0 = SCReLU |
| 22 | 2 | `qa` / `qb` / `eval_scale` | 255 / 64 / 400, at 22, 24 and 26, read unsigned |
| 28 | 4 | `reserved` | zero — refused otherwise, so a later version can spend these bytes safely |
| 32 | 8 | `king_bucket_map_crc` | `0x0A20060D4E9586E0` — CRC-64/XZ of the map's 64 entries as `uint8` |
| 40 | 8 | `payload_bytes` | 12617760 |
| 48 | 8 | `payload_crc` | CRC-64/XZ over the payload |
| 56 | 8 | `header_crc` | CRC-64/XZ over bytes 0..55 |

The checksum is **CRC-64/XZ** (polynomial `0x42F0E1EBA9EA3693`, reflected, init and final XOR all ones);
every implementation reproduces the check value `0x995DC9BBDF1939FA` for the nine ASCII bytes `123456789`,
which `nnuecheck` asserts before it looks at a net.

The loader makes two checks, in this order, and **refuses on any mismatch without falling back**: the
header (magic, `header_crc`, then every architecture field against the engine's own compile-time constants,
then `king_bucket_map_crc` against the engine's own map), then the payload (its length exactly as claimed,
nothing appended, then `payload_crc`). The map's digest is there because `king_buckets` matching while the
map's *contents* differed used to load cleanly and evaluate from the wrong weight bank — which is also why
turning mirroring on moves the format version rather than only a flag.

Nets are stamped by **`tools/stamp_net.py`**, which writes the header from `trainer/rnnue1.py`'s own
constants and never from an argument, so a payload of the wrong shape gets a header every engine refuses
instead of one that describes it. Never hand-edit a net file.

The retired `ZNNUE5`, `ZNNUE4` and `ZNNUE3` formats are **no longer read** (2026-10-27). They carried an
8-byte magic and nothing else, which is the exact failure this container exists to prevent; every net that
ever existed in `ZNNUE4` or `ZNNUE3` was 512 wide and so was already refused on size by the shipped
1024-wide engine. A `ZNNUE5` net re-stamps losslessly — the payload layout is identical, output bias
included — with `python3 tools/stamp_net.py <net>`.

The integer forward (identical in `integer_eval` of `trainer/features.py` and `src/nnue.c`; `b` = the
position's output bucket):

```
acc = FT_bias + Σ FT_weight[active feature]        (per perspective; int16 columns, int64 accumulation)
h   = clamp(acc, 0, QA)                            (SCReLU input clamp, per perspective)
out = Σ (h · OUT_w[b]) · h                         (own half + opponent half; the ·h squares → SCReLU)
out = out / QA + OUT_bias[b]
cp  = out × EVALUATION_SCALE / (QA × QB)           (all divisions truncate toward zero, matching C `/`)
```

The export step prints how many transformer weights saturate `int16` (should be 0 for a healthy net).

---

## 3. Design tradeoffs

- **Perspective net over a plain 768→N.** Evaluating "from the side to move's view" halves what the net has
  to learn (it never has to represent both orientations) and makes the accumulator reusable across plies.
  Standard and strongly positive.
- **King-input buckets (768×8).** Conditioning features on the perspective's own king square lets the net
  learn king-safety-dependent piece values. Worth **+21 Elo at fixed depth**, but it costs ~6% speed (a king
  move that changes bucket forces an accumulator refresh) and needs *more data* to pay for that speed hit —
  it only overtakes the plain 512-net once the dataset is large (see §9).
- **Material output buckets (1024→8 heads).** Selecting the output head by total piece count lets each
  game phase get its own 1024→1 mapping for almost no cost (the buckets share the whole feature
  transformer; only the final dot product changes). An early attempt (a `ZNNUE2` net) was neutral, but retried on
  the 1.4B-position dataset it was worth **+14.7 Elo at fixed depth** (`zenith-ob2`) — bucket ideas can be
  data-starved rather than wrong.
- **Horizontal king mirroring.** Chess has no left/right asymmetry in the rules, so a perspective whose king
  sits on files e–h is mirrored (`square ^ 7`) onto files a–d before feature indexing. Halves the king-
  placement space the net must model: same file size, twice the effective data per bucket. The cost is an
  accumulator refresh when a king crosses the d/e boundary (served by the same finny refresh cache as
  bucket changes).
- **SCReLU over clipped ReLU.** `clamp(x,0,1)²` gives a stronger net than plain clipped ReLU at the same
  size and is cheap as an integer kernel (`(h·w)·h`). The clamp bound is exactly `QA`, so quantisation and
  activation share one constant.
- **Hidden size 1024.** Bigger accumulators improve eval but cost proportional per-node time and training
  memory. 512 was the shipped balance until 1.4B positions saturated it: at that scale 1024 gained +52 Elo
  at fixed depth and **+14.7 at time control** (cap1) — the eval gain outran the doubled per-node cost. The trainer uses an `EmbeddingBag(mode="sum")` feature
  transformer so a batch never materialises a `[batch, 32, hidden]` intermediate — this is what keeps larger
  hidden sizes trainable on an 8 GB GPU.
- **WDL blend (`--wdl-lambda`).** The training label is `λ·game_result + (1−λ)·sigmoid(score/400)`: λ→1
  trusts the eventual game outcome, λ→0 trusts the search score of the position. Mid-range values train the
  cleanest net; the shipped net used λ=0.3.
- **Streaming vs monolithic training.** Holding the whole featurised dataset in RAM caps out around ~230M
  positions on a 62 GB box. Past that, per-shard `.npz` caches are streamed one at a time (`--shard-dir`),
  which is how the shipped net trained on 1.4B positions. Shard-order shuffling (reseeded per epoch) plus a
  full in-shard permutation is effectively i.i.d. at ~95M-position shards.
- **The eval is a faithful executor — fix data, not the loader.** If a *correct* net plays weakly, the cause
  is eval noise (data quantity/quality), not the engine: minimax amplifies leaf-eval noise. The lever is
  more/cleaner data and better training, not engine code.

---

## 4. The contract & verification gates

`trainer/features.py` is the **single source of truth** for feature indexing and quantisation; `src/nnue.c`
must reproduce `feature_index`, `king_bucket`, and `integer_eval` byte-for-byte. Two gates enforce this and
are never skipped:

- **0-cp trainer parity** — `trainer/verify.py` loads a `.nnue`, evaluates a FEN set with the Python
  reference `integer_eval`, runs the same FENs through `./build/zenith nnueeval <net>`, and requires
  `max|engine − reference| == 0 cp`. Any nonzero diff means the loader or forward diverged from the export
  contract.
- **Incremental == refresh** — `./build/zenith nnuecheck <net>` walks a perft-like tree and checks the
  incrementally-maintained accumulator against a full refresh at every node, bit-identical (guards the
  finny-cache / king-bucket refresh logic).

Also worth a spot-check: `eval(pos) == eval(colour-mirror of pos)` — the perspective symmetry must hold.
Every net change is then **SPRT-gated** like any strength change (see §7).

---

## 5. Training setup

- **Hardware used:** NVIDIA RTX 4070 Laptop GPU (8 GB) + 32 CPU cores. Datagen and SPRT are CPU-bound;
  training is the GPU job — they can run concurrently.
- **Python env** (venv is `.venv`, git-ignored): `tools/setup-dev.sh --training` (Windows:
  `tools\setup-dev.ps1 -Training`) makes it with uv - torch from PyTorch's CUDA 13.0 index on Linux and Windows, PyPI's
  Metal build on macOS - and reports whether torch sees the GPU; by hand:
  ```bash
  python3 -m venv .venv && . .venv/bin/activate
  pip install -r trainer/requirements.txt      # torch>=2.3 (CUDA build on Linux) + numpy>=1.26
  ```
- The trainer imports `features.py` as a top-level module, so run it with `PYTHONPATH=trainer` (or from
  inside `trainer/`). `data/` and `nets/` are git-ignored; commit a chosen release net with `git add -f`.

---

## 6. Training a new network — step by step

There are two data sources; both produce the same `fen;stm_score_cp;wdl` text the trainer consumes, so the
train / verify / SPRT steps afterwards are identical.

### Data, option A — Zenith self-play (fully independent)

Generate self-play games at a fixed node budget, fanned across all cores. Each worker is an independent
`./build/zenith-datagen` process with its own seed writing one shard; only quiet, not-yet-decided positions are
emitted (one record per ply), labelled with the eventual game result as WDL.

```bash
make datagen                                     # build ./build/zenith-datagen first
tools/datagen_parallel.sh 20000 data/run1 32 5000 8
#                          │     │        │  │    └ opening plies (random legal plies before recording)
#                          │     │        │  └ nodes/move budget (go nodes N)
#                          │     │        └ workers (default: nproc)
#                          │     └ output dir (one shard_NN.txt per worker)
#                          └ games per worker
```
Optionally pass a book EPD as arg 6 and a net as arg 7 (`… 8 book.epd nets/prev.nnue`) to label with a net
already in the loop — the net-in-the-loop bootstrap that lifts data quality above the HCE teacher.

### Data, option B — the public PlentyChess dataset at scale (how the nets up to `zenith-cap2` trained)

The nets up to and including `zenith-cap2` trained on positions from the public self-play dataset
generated by **PlentyChess**, by Yoshie2000. **That data is no longer used**: do not convert,
re-download or extend its lineage. Both those nets and pawnstar's shipped v12 derive from it, and
retraining the shared evaluator off a clean pool is why the shared NNUE work exists. The net shipped from
3.0.0 trained on Leela Chess Zero self-play data instead — see §9.
Convert each bulletformat shard to Zenith's text format, subsampling with a stride for a diverse slice:

```bash
./build/zenith-bullet2text data/bullet_raw/shard00.data data/bullet/shard00.txt 0 4
#                    └ input (bulletformat)         └ output text          │ └ stride (keep 1 in 4)
#                                                                          └ max records (0 = all)
```

### Train

**Monolithic** (whole dataset in RAM, ≲230M positions). `--cache` writes/loads a featurised `.npz` so
re-runs skip featurisation:

```bash
PYTHONPATH=trainer python trainer/train.py \
    --data 'data/run1/shard_*.txt' --out nets/zenith-new.nnue --cache data/run1.npz \
    --epochs 30 --batch-size 16384 --lr 1e-3 --wdl-lambda 0.5 --hidden-size 512 --device cuda
```

**Streaming** (past the RAM ceiling — required at the ~1.4B-position scale). First featurise each text shard
to a `.npz` (chunked, low-RAM, safe to run several in parallel), then stream-train over the directory (the
last shard is held out as the validation probe):

```bash
# featurise shards -> data/shards/*.npz  (parallelise across shards; resumable, skips existing)
for s in data/bullet/shard*.txt; do
    PYTHONPATH=trainer python trainer/train.py --featurise-shard "$s" "data/shards/$(basename "${s%.txt}").npz"
done

PYTHONPATH=trainer python trainer/train.py \
    --shard-dir data/shards --out nets/zenith-new.nnue \
    --hidden-size 512 --epochs 8 --batch-size 32768 --lr 1.2e-3 --wdl-lambda 0.3 --device cuda
```

Training prints per-epoch `train`/`val` loss (AdamW + cosine LR decay). Falling val loss that then flattens
is the signal to stop; rising val loss is overfit. Output: `nets/zenith-new.nnue` plus a `.pt` checkpoint.

### Verify (never skip)

```bash
PYTHONPATH=trainer python trainer/verify.py --net nets/zenith-new.nnue --fens data/run1/shard_01.txt --count 2000
./build/zenith nnuecheck nets/zenith-new.nnue
```
Both must pass (`0 cp`, `0 mismatches`) before the net is worth testing for strength.

---

## 7. Deciding whether it is better (SPRT)

A verified net is only *correct*, not necessarily *stronger*. Gate it with a self-play SPRT via
`tools/sprt.sh` (fastchess). `CAND_NET` / `BASE_NET` set each side's `EvalFile`; an unset side uses its embedded net.
**`OPENINGS` is required** — no opening set ships in this repo, and `books/`'s Polyglot `.bin` files are not
one (fastchess wants EPD or PGN).

```bash
# new net vs the current release net
OPENINGS=/path/to/openings.epd     CAND_NET=nets/zenith-new.nnue BASE_NET=nets/reckless-lc0-runf-sb870.nnue tools/sprt.sh
# new net vs the embedded net (BASE_NET unset)
OPENINGS=/path/to/openings.epd CAND_NET=nets/zenith-new.nnue tools/sprt.sh
```

The script pins `Threads=1`, `Hash` (16 MB, `$HASH`) and `OwnBook=false` on both engines, and prints them
beside the result, so a measurement does not depend on a UCI default or on the machine's core count. It
defaults `CONCURRENCY` to one game per physical core less one.

Fixed-depth matches isolate eval quality from NNUE's speed cost. **Self-play gains only partially transfer
to a much stronger reference** (an eval win of +60 in self-play was ~+20 vs the pawnstar reference), so SPRT
self-play deltas overstate the gain against a stronger opponent, and speed wins transfer far better than
eval wins. The full experiment log with every result and lesson is in
`.claude/memory/zenith-eval-experiments.md`.

---

## 8. Engine integration (already implemented)

`src/nnue.c` holds the loader, feature indexing, and the integer forward:

- **Incremental accumulator** — embedded in `Position`; the put/remove/move primitives apply feature-column
  deltas as moves are made, so copy-make carries a ready accumulator.
- **King-bucket refresh via a finny cache** — a king move that changes a perspective's bucket rebuilds that
  perspective from a thread-local per-`(perspective, bucket)` cached accumulator (plus the board it was built
  from), applying only piece diffs; a net-generation counter invalidates the cache when a new net loads.
- **AVX2 forward** — vectorised accumulator column add/sub and the SCReLU output dot, with a scalar fallback.
- **Shared lockless eval cache** — memoises `evaluate()` results (biggest win: qsearch stand-pat).

Load a net at runtime with `setoption name EvalFile value nets/<name>.nnue`; unset (or a load
failure) ⇒ the build-time-embedded net.

---

## 9. Current release net

`nets/reckless-lc0-runf-sb870.nnue` — king-bucketed, horizontally-mirrored **768×8 → 1024 → 8 heads**
SCReLU in the shared `RNNUE1` container, trained **entirely on Leela Chess Zero self-play data**. It is
embedded in every binary at build time (`Makefile NET`) from **3.0.0** onward. It is the only net in the
tree: `zenith-cap1.nnue` and `zenith-cap2.nnue` were removed at 3.0.0 (see below), and git history
preserves them.

### Provenance of earlier networks

Zenith networks up to and including **zenith-cap2** were trained on positions from a public self-play
dataset generated by **PlentyChess**, by Yoshie2000 — up to 2.15 billion positions. That data is no
longer used: the network shipped from **3.0.0** onward is trained entirely on Leela Chess Zero self-play
data.

The attribution is kept here deliberately. Those networks were measurably stronger for having been
trained on that data, and the record should show where they came from.

`zenith-cap1.nnue` and `zenith-cap2.nnue` are no longer in the tree, so no release ships a net derived
from that data. Zenith's own `datagen` generator is option A above and can train a net, but the shipped
nets took option B.

The 3.0.0 net is not a product of Zenith's own trainer either, which is why it does not carry a `zenith-`
name: it came from the shared Reckless NNUE training pipeline, at the shape and in the `RNNUE1` container
every Reckless engine now reads, and `runf-sb870` names the run and checkpoint it came from. Zenith's
PyTorch trainer in `trainer/` is unchanged and still builds nets of this shape.

Progression of the eval (each step SPRT-gated). Every measurement below stands exactly as it was made;
everything from `pc2` to `cap2` is on the PlentyChess data, so those deltas measure architecture and
volume within that lineage, not against the 3.0.0 net:

| net | arch / data | result |
|---|---|---|
| pc2 | 768→512, 190M | baseline |
| kb2 | + king buckets, 650M | **+57 Elo** over pc2 (self-play) |
| kb3 | + 1.4B positions | +8 over kb2 — data returns diminishing |
| ob2 | + material output buckets | +14.7 over kb3 (fixed depth) |
| km1 | + horizontal king mirroring | +14.3 over ob2 (fixed depth) |
| cap1 | hidden 512 → **1024** | +52 fixed depth, **+14.7 timed** over km1 |
| cap2 | + 2.15B positions | **+10** over cap1 (fixed depth); val loss 0.013746 |
| `reckless-lc0-runf-sb870` | same shape, **Leela Chess Zero** data | strength in Zenith **not yet measured** — see below |

Superseded nets are deleted from the tree once the shipped engine can no longer load them (git history
preserves them; the `RNNUE1` header names every architecture field, so a mismatch is refused by name
rather than inferred from a file's length). `zenith-cap1.nnue` and `zenith-cap2.nnue` were removed for a
different reason — the 3.0.0 engine loads both perfectly well, they being the same shape: they are
PlentyChess-derived, and no release ships such a net.

**The new net's strength in Zenith is unmeasured, and the swap did not wait on it** — removing the data
dependency is the point, so the measurement tells us what the change cost or gained, not whether to make
it. The net loads unchanged (same architecture, same container, `nnuecheck` and `trainer/verify.py` both
clean) and it moves the bench node signature from 1,390,606 to **1,653,778**, which is an eval change and
says nothing about strength. Figures measured for this net in **pawnstar**, against *pawnstar's* old net,
are not Zenith results and do not transfer.

The ~**+135 Elo** single-thread at 8+0.08 over the reference opponent (pawnstar), bookless on both sides,
was measured with `zenith-cap2` and stands as made (pawnstar's defaults `Threads=32`/`OwnBook=true` must
be overridden or the measurement is meaningless); it has not been re-measured with the 3.0.0 net.
