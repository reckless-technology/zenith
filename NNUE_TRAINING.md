# Zenith NNUE — training plan & GPU handoff

This is the resume-here doc for Phase 4 (the ~+500–700 Elo jump). It is written so that on an
NVIDIA-GPU machine we can pick up and train Zenith's own net from scratch, independent of pawnstar.

Everything below plugs into the existing `evaluate()` seam in `src/eval.cpp` — no search changes needed to
adopt NNUE; the HCE stays as a fallback and as the eval used during data generation.

---

## 1. Network architecture (v1 — deliberately simple, strong, trainable in hours)

A **perspective network** (the "simple NNUE" that most 3000+ hobby engines start from):

```
inputs:   768 per side  =  6 piece types × 2 colours × 64 squares, from the side-to-move's POV
          (black-to-move mirrors square vertically and swaps colours)
king      the perspective's own king square selects 1 of 8 KING BUCKETS (4 file-pairs × 2 board-halves),
buckets:  offsetting its 768 block → 768×8 = 6144 feature-transformer rows
FT:       6144 → 512   (feature transformer; one accumulator per side)
concat:   [own(512), opp(512)] = 1024
activate: SCReLU(x) = clamp(x, 0, 1)²        (screlu is stronger than clipped-relu; matches the engine kernel)
output:   1024 → 1   (single scalar), then dequantize to centipawns
```

The shipped net (`ZNNUE3`) is this king-bucketed 768×8→512 perspective net. History: plain 768→512 (`ZNNUE1`)
was the v1 baseline; king buckets add real eval strength (+21 Elo at fixed depth) but need a bigger data set
to beat their ~6% speed cost — trained on **650M** PlentyChess positions the king-bucket net is **+57 Elo**
over the 190M 512-net. (An 8-head output-bucket variant `ZNNUE2` was tried and shelved: neutral Elo.)

### Quantization & on-disk format (fixed contract: trainer export == engine loader)

```
QA = 255   (feature/accumulator scale)     QB = 64   (output-weight scale)     eval_scale = 400
FT weights  : round(w * QA)      -> int16   [6144][512]   (king-bucketed: 8 × 768 rows)
FT bias     : round(b * QA)      -> int16   [512]
OUT weights : round(w * QB)      -> int16   [1024]
OUT bias    : round(b * QA*QB)   -> int32   [1]
```

**File `zenith-<tag>.nnue`** (little-endian): 8-byte magic `"ZNNUE3\0\0"`, then the four arrays back-to-back
in the order above. The engine loader (`src/nnue.cpp`) reads it directly. Forward pass:
`acc = FT·features + FT_bias` (per side, maintained incrementally with king-bucket refreshes), then
`out = Σ screlu(acc_own,acc_opp) · OUT_w + OUT_bias`, `eval_cp = out / (QA*QB) * eval_scale / QA` (fold the
constants into one final divide; verify against the trainer to 0 cp on a fixed FEN set).

---

## 2. Data generation (self-play; independent of any external dataset)

Add a `datagen` mode to the engine (new `src/datagen.cpp`, invoked `./zenith datagen <games> <out.txt>`):

- Play self-play games from **random UHO openings** (or 8 random legal plies) at a **fixed node budget**
  (e.g. `go nodes 5000`), single-thread, hash cleared per game.
- For every position that is **quiet** (not in check, best move not a capture/promo) and not already decided
  (|score| < ~1500cp), emit one record: `fen ; stm_relative_score_cp ; wdl` where `wdl ∈ {1.0, 0.5, 0.0}` is
  the game result from the side-to-move's POV.
- Skip the first ~8 plies (opening noise) and positions right after a capture. One record per position.
- Target **100M–300M positions** for v1 (a few CPU-hours × many cores; the SPRT box's cores are fine — this
  step is CPU-bound, not GPU). Shard to multiple files and shuffle.

Output format is plain text `fen;score;wdl` for portability; a 10-line converter turns it into the trainer's
tensor batches (or into `bulletformat`/`marlinformat` if using bullet).

---

## 3. Trainer (GPU)

**Primary: a self-contained PyTorch trainer** (`trainer/train.py`, ~200 lines — independent, full control):

- Model: `FT = nn.EmbeddingBag(6144+1, 512, mode="sum")` (memory-efficient sparse feature transformer, one
  row per king-bucketed feature + a zero pad row); forward builds both perspectives from the sparse feature
  indices, concatenates, `SCReLU`, `out = nn.Linear(1024, 1)`.
- Batching: parse `fen;score;wdl`; featurize to sparse **king-bucketed** index lists per side. Two paths:
  monolithic `--cache` (whole dataset in RAM, ≤~230M positions on a 62GB box) or **`--shard-dir` streaming**
  (one ~95M-position `.npz` shard in RAM at a time — how the shipped 650M-position net trained). Precompute
  shard caches with `--featurise-shard TEXT NPZ` (chunked, low-RAM, run several in parallel). Batches 16k–65k.
- Loss: `MSE( sigmoid(pred/eval_scale), λ·wdl + (1−λ)·sigmoid(score/eval_scale) )`, λ≈0.5 → anneal toward WDL.
- Optim: AdamW, lr 1e-3 cosine-decayed, ~30–100 epochs over shuffled data. A first net trains in **1–4 h** on
  a modern GPU.
- Export: quantize per §1 and write `zenith-<tag>.nnue`.

**Faster alternative: [bullet](https://github.com/jw1912/bullet)** (CUDA, the de-facto NNUE trainer). Convert
datagen output to `bulletformat::ChessBoard`, train the same 768→512 perspective arch, export, and adapt the
quantization to the format above. Bullet is ~10–100× faster to a strong net; use it once the PyTorch path is
validated end-to-end, or straight away if we want speed.

---

## 4. Engine integration (`src/nnue.{h,cpp}`)

1. **Loader**: read the `.nnue` file into aligned int16 arrays.
2. **Feature index**: `idx(perspective, colour, pt, sq)` with the mirror/colour-swap for the black POV.
3. **Incremental accumulator**: keep `Accumulator{ int16 own[512], opp[512] }` on the search stack; on
   `make_move`, apply the piece deltas (add/sub the moved/captured/promoted feature columns) instead of
   recomputing — this is what makes NNUE cheap. Full refresh on a king move (needed once king buckets land).
4. **Forward**: SCReLU + int16/int8 output dot. Start scalar (correct), then add an **AVX2 kernel**
   (`_mm256_madd_epi16`, the same shape as pawnstar's `outputDotInt8`) for ~8× on the hot path.
5. **Seam**: `evaluate()` calls `nnue::eval(pos)` when a net is loaded (UCI `EvalFile`), else the HCE.

**Verification gates** (mirror the perft discipline):
- `nnue::eval` (int16) == the trainer's float eval to **0 cp** on a fixed FEN set (a `TestNNUEReference`).
- incremental accumulator == full refresh, bit-identical, on every node of a perft-like walk.
- AVX2 dot == scalar dot, bit-identical.
- **SPRT NNUE-net vs HCE** (`tools/sprt.sh`) — expect a large positive (+500–700); iterate net size/data.

---

## 5. Resume checklist

Done (v1 pipeline built + validated 2026-07-18):
```
[x] make ; ./zenith perft                                   # core builds + correct
[x] src/datagen.cpp (§2) + tools/datagen_parallel.sh        # ./zenith datagen; fan over cores
[x] trainer/train.py + trainer/features.py (§3, PyTorch)    # quantised .nnue export
[x] src/nnue.{h,cpp} (§4): loader + SCReLU integer forward  # FULL REFRESH (not yet incremental)
[x] verification gate: trainer/verify.py == 0 cp            # engine int eval == trainer, bit-identical
[x] wire evaluate() -> nnue when EvalFile set; UCI EvalFile option; nnueeval CLI
[x] SPRT harness (tools/sprt.sh, fastchess): NNUE vs HCE, cross-engine
```

Open (the climb — this is now a data/training program, not engine code):
```
[ ] cut eval NOISE: the pilot net (4M pos, HCE self-play) LOSES to HCE (~-325 Elo) because it correlates
    only 0.92 with its labels (~267cp noise) and minimax amplifies leaf noise. Fix: more + cleaner data
    (deeper search labels, dedup), more/better training (lower wdl-lambda, more epochs), larger net.
[ ] incremental accumulator (make full-refresh ~3x faster) + later an AVX2 kernel
[ ] once NNUE beats HCE: SPRT vs pawnstar; then bigger net + more data; v2 HalfKA + king buckets
```

## State (2026-07-19)

- `main` = perft-correct classical core + PVS search + **the full NNUE pipeline** (datagen incl. book &
  net-in-the-loop, PyTorch trainer, quantised AVX2 loader), all verified (0 cp engine-vs-trainer,
  colour-mirror symmetric, tactics found). Current release net: `nets/zenith-v6.nnue`.
- **Net progression (fixed-depth-8 Elo vs the PeSTO HCE):** pilot −325 (underfit/noisy) → v2 −38 → v4 (46M)
  **+139** → v5 (72M, self-play) **+301 (+166 timed — wins timed)** → v6 (78M window) **+284, +60 over v5**.
- **Key results & lessons:** more data is the dominant lever; **self-play (net-in-the-loop) compounds**
  (v5-vs-v4 +211) but with **diminishing returns** (v6-vs-v5 +60); 512 hidden is the sweet spot (768
  overfits); int16+AVX2 refresh nearly doubled nps (641k→1.08M). vs pawnstar: **−489 → −377/−385** timed.
- **Strategic read:** the remaining gap to pawnstar is now **speed + search + parallelism bound, not eval
  quality** (v6's better eval did not close the timed pawnstar gap). Next levers, each SPRT-gated:
  incremental accumulator (or make/unmake) for speed, singular extensions / IIR, Lazy SMP + lockless TT.
- Harness: `tools/sprt.sh` uses **fastchess**; `~/pawnstar_nnue/openings.epd` is the default book. SPRT is
  CPU-bound, training is the GPU job — they run concurrently.
```
