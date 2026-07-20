---
name: pawnstar-gap-benchmarks
description: Zenith-vs-pawnstar Elo gaps at various configs (the reference-opponent benchmark)
metadata: 
  node_type: memory
  type: project
  originSessionId: 221925ef-3c7d-4275-96ac-0cd7362cb511
---

Zenith's strength is benchmarked against pawnstar C++ (the reference opponent; user wants Zenith stronger). Gaps measured with fastchess, tc=8+0.08, plentychess openings:

- **Single-thread, pc2 net (512, 190M plentychess):** −238 ± 52 Elo vs pawnstar (timed). Earlier NNUE+speed+search work had already halved an initial −489 gap to about −211.
- **8 threads each (Lazy SMP), pc2 net:** −42 ± 48 Elo vs pawnstar (measured 2026-07-20). CI includes zero → near-even at equal threads. Zenith's Lazy SMP scales better than pawnstar's, closing ~196 Elo of the single-thread gap.

Takeaway: the remaining single-thread deficit is dominated by **eval quality** (pawnstar's net is stronger), not search. Hence the ongoing eval experiments (output buckets, king buckets). Net-quality work is the highest-leverage path to overtaking pawnstar single-threaded. See [[nnue-training-pipeline]] and [[zenith-eval-experiments]].
