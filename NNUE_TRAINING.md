# Zenith NNUE — architecture & training guide

Zenith evaluates positions with its own king-bucketed NNUE, trained by the pipeline in this repository.
Nothing here is shared with any other engine: Zenith has its own network architecture, its own trainer
(`trainer/`), its own quantised file format (`ZNNUE3`), and its own trained weights. This document is the
reference for how the network is built and how to train a new one.

The pipeline is four stages:

```
   data                         trainer (PyTorch/CUDA)              engine (C)
 ┌─────────────────┐  text     ┌──────────────────────┐  .nnue   ┌───────────────────┐
 │ ./build/zenith datagen│ ────────▶ │ trainer/train.py     │ ───────▶ │ src/nnue.c loader  │
 │  (self-play)    │  fen;     │  featurise → SCReLU  │  ZNNUE3  │  + integer forward │
 │  OR             │  score;   │  net → quantise      │          │  behind evaluate() │
 │ ./build/zenith        │  wdl      │                      │          │                    │
 │  bullet2text    │           └──────────────────────┘          └───────────────────┘
 │  (public data)  │                     │                                 │
 └─────────────────┘                     └────── trainer/verify.py ────────┘
                                          engine int eval == trainer ref (0 cp)
```

`evaluate()` in `src/eval.c` is the single seam: it calls the NNUE forward pass when a net is loaded (UCI
`EvalFile`) and falls back to the hand-crafted PeSTO evaluation otherwise. No search change is needed to
adopt or swap a net.

---

## 1. Network architecture

A **perspective network** with **king-input buckets** — the design most strong hobby engines converge on:

```
per-perspective inputs : 768  =  2 colors × 6 piece types × 64 squares   (own pieces first)
king buckets           : the perspective's OWN king square selects 1 of 8 buckets
                         (4 file-pairs {a/b, c/d, e/f, g/h} × 2 board-halves {ranks 1-4, 5-8}),
                         offsetting the 768 block → 768 × 8 = 6144 feature-transformer rows
feature transformer    : 6144 → 512     (one 512-wide accumulator per perspective)
concatenate            : [own(512), opponent(512)] = 1024
activation             : SCReLU(x) = clamp(x, 0, 1)²
output                 : 1024 → 1  (single scalar logit), dequantised to centipawns
```

Two accumulators are maintained, one per perspective (side-to-move "own" and the opponent). Black's
perspective mirrors the board vertically (`square ^ 56`) and swaps the own/opponent colour halves, so the
network only ever learns "from the side to move's point of view."

The feature-index convention lives in **`trainer/features.py`** (`feature_index`, `king_bucket`) and is the
single source of truth; `src/nnue.c` reproduces it exactly. Feature layout within one king bucket:

```
index = king_bucket(rel_king) × 768  +  rel_color × 384  +  piece_type × 64  +  rel_square
        rel_color  = 0 for the perspective's own pieces, 1 for the opponent's
        rel_square = square (White perspective)  or  square ^ 56 (Black perspective)
        piece_type = 0..5  (pawn..king, 0-based on the wire)
```

Hidden size is a knob (`--hidden-size`, default and shipped value **512**); everything else above is fixed
by the format.

---

## 2. Quantisation & on-disk format (`ZNNUE3`)

The trainer exports a fixed-point `int16` network; the engine loads it directly and does an all-integer
forward pass (a float `.pt` checkpoint is saved alongside for inspection only).

```
QA = 255   feature-transformer weight / accumulator scale
QB = 64    output-weight scale
EVALUATION_SCALE = 400   logit → centipawn scale (must equal the trainer loss scale)

FT weights : round(w × QA)      → int16   [6144][512]   (feature-major, king-bucketed: 8 × 768 rows)
FT bias    : round(b × QA)      → int16   [512]
OUT weights: round(w × QB)      → int16   [1024]         (own half [0,512) then opponent half [512,1024))
OUT bias   : round(b × QA × QB) → int32   [1]
```

**File `nets/zenith-<tag>.nnue`** (little-endian): the 8-byte magic `ZNNUE3\0\0`, then the four arrays
back-to-back in the order above. The integer forward (identical in `integer_eval` of `trainer/features.py` and
`src/nnue.c`):

```
acc = FT_bias + Σ FT_weight[active feature]        (per perspective; int16 columns, int64 accumulation)
h   = clamp(acc, 0, QA)                            (SCReLU input clamp, per perspective)
out = Σ (h · OUT_w) · h                            (own half + opponent half; the ·h squares → SCReLU)
out = out / QA + OUT_bias
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
  it only overtakes the plain 512-net once the dataset is large (see §9). An 8-way **output**-bucket variant
  (`ZNNUE2`) was tried and **shelved** — neutral Elo for the added complexity.
- **SCReLU over clipped ReLU.** `clamp(x,0,1)²` gives a stronger net than plain clipped ReLU at the same
  size and is cheap as an integer kernel (`(h·w)·h`). The clamp bound is exactly `QA`, so quantisation and
  activation share one constant.
- **Hidden size 512.** Bigger accumulators (e.g. 1024) improve eval but cost proportional per-node time and
  training memory; 512 is the shipped balance. The trainer uses an `EmbeddingBag(mode="sum")` feature
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
- **Python env** (venv is `.venv`, git-ignored):
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
`./build/zenith datagen` process with its own seed writing one shard; only quiet, not-yet-decided positions are
emitted (one record per ply), labelled with the eventual game result as WDL.

```bash
make                                             # build ./build/zenith first
tools/datagen_parallel.sh 20000 data/run1 32 5000 8
#                          │     │        │  │    └ opening plies (random legal plies before recording)
#                          │     │        │  └ nodes/move budget (go nodes N)
#                          │     │        └ workers (default: nproc)
#                          │     └ output dir (one shard_NN.txt per worker)
#                          └ games per worker
```
Optionally pass a book EPD as arg 6 and a net as arg 7 (`… 8 book.epd nets/prev.nnue`) to label with a net
already in the loop — the net-in-the-loop bootstrap that lifts data quality above the HCE teacher.

### Data, option B — public PlentyChess data at scale (how the shipped net trained)

The shipped net trained on the **public PlentyChess bulletformat dataset** (independent of any engine).
Convert each bulletformat shard to Zenith's text format, subsampling with a stride for a diverse slice:

```bash
./build/zenith bullet2text data/plenty_raw/shard00.data data/plenty/shard00.txt 0 4
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
for s in data/plenty/shard*.txt; do
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
`tools/sprt.sh` (fastchess). `CAND_NET` / `BASE_NET` set each side's `EvalFile`; an unset side uses the HCE.

```bash
# new net vs the current release net
CAND_NET=nets/zenith-new.nnue BASE_NET=nets/zenith-kb3.nnue tools/sprt.sh
# new net vs HCE (BASE_NET unset)
CAND_NET=nets/zenith-new.nnue tools/sprt.sh
```

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

Load a net at runtime with `setoption name EvalFile value nets/zenith-<tag>.nnue`; unset ⇒ HCE.

---

## 9. Current release net

`nets/zenith-kb3.nnue` — king-bucketed **768×8 → 512** SCReLU, trained with the streaming trainer on **1.4B**
PlentyChess positions. Progression of the eval:

| net | arch / data | result |
|---|---|---|
| pc2 | 768→512, 190M | baseline |
| kb2 | + king buckets, 650M | **+57 Elo** over pc2 (self-play) |
| kb3 | + 1.4B | +8 over kb2 — **data returns now diminishing** |

With this net Zenith beats its reference opponent (pawnstar) at every tested configuration — **+78 Elo
single-thread and +107 at 8 threads** (TC 8+0.08, single-threaded and bookless on both sides; pawnstar's
defaults `Threads=32`/`OwnBook=true` must be overridden or the measurement is meaningless). The
highest-leverage remaining lever is eval quality — more/cleaner data — not engine changes.
