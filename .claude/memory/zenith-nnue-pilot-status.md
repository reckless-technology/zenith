---
name: zenith-nnue-pilot-status
description: NNUE pipeline validated end-to-end; the pilot net loses to HCE due to eval noise, not a bug
metadata:
  type: project
---

The Zenith NNUE pipeline (datagen → PyTorch train → quantised `.nnue` → engine full-refresh eval) is
CORRECT and validated: the 0cp gate passes (engine integer eval == trainer reference, bit-identical over
thousands of positions), the net is perfectly symmetric (eval == colour-mirror eval, max diff 0),
material signs/magnitudes are right, and the search finds tactics (free queen, back-rank mate).

The first pilot net (4.1M positions, 512 hidden, 30 epochs, wdl-lambda 0.5) LOSES to the PeSTO HCE by
~−325 Elo at fixed depth (−572 at 8+0.08). Root cause is EVAL NOISE, not a bug: the net correlates only
0.92 with the HCE scores it was trained on (~267cp residual error std), and minimax amplifies leaf-eval
noise into blunders (e.g. hanging a queen). Full-refresh NNUE also runs ~3× slower (~640k vs 1.8M nps).

**How to apply:** to beat HCE (then pawnstar) the net must be much less noisy — more data (scaling via
`data/v2` at 6000 nodes), longer/better training (lower wdl-lambda, more epochs), a larger net, and later
an incremental accumulator for speed. Do NOT re-debug the pipeline as broken; it is correct.
Related: [[zenith-goal-independence]] [[training-setup]]
