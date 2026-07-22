---
name: pawnstar-gap-benchmarks
description: Zenith-vs-pawnstar Elo gaps at various configs (the reference-opponent benchmark)
metadata: 
  node_type: memory
  type: project
  originSessionId: 221925ef-3c7d-4275-96ac-0cd7362cb511
---

Zenith's strength is benchmarked against pawnstar C++ (the reference opponent; user wants Zenith stronger). Gaps measured with fastchess, tc=8+0.08, plentychess openings.

**IMPORTANT — always match `-concurrency` when comparing gap numbers.** Concurrency changes the timed result a lot: pc2 vs pawnstar was −238±52 at conc 30 but −277±44 at conc 8 (same net, same TC). Cross-run comparisons at different concurrency are meaningless.

Single-thread, matched conditions (conc 8):
- **pc2 (512, 190M plentychess):** −277.5 ± 43.6 vs pawnstar (2026-07-21).
- **kb2 (king buckets, 650M plentychess):** −257.6 ± 40.7 → kb2 closes ~+20 Elo (eval gain, partial transfer).
- **kb2 + correction history + legality oracle (current ./zenith):** **−140.4 ± 34.1** (2026-07-22) → the full
  stack closes **~+137 Elo**; the single-thread gap has roughly HALVED from −278 to −140.
- **kb2 vs pc2 self-play: +63.2 ± 29.7** (consistent with the +57 SPRT that shipped kb2).

Session Elo (self-play SPRT): kb2 +57, correction history +10, legality oracle +66 (~+133 cumulative). The
legality-oracle speed win transferred to pawnstar near 1:1 (speed helps vs any opponent), unlike eval gains
(~1/3 transfer) — which is why the full-stack gap closure (~+137) exceeds kb2's eval-only +20.

8 threads each (Lazy SMP), pc2 net: −42 ± 48 vs pawnstar (conc 3, 2026-07-20) — near-even at equal threads; Zenith's SMP scales better than pawnstar's. (Re-measure with kb2 at matched conc for a clean number.)

**Key lesson — self-play gains only partially transfer to a much stronger opponent.** kb2 is +63 vs pc2 in
self-play but only ~+20 better against pawnstar (~1/3 transfer). SPRT self-play deltas OVERSTATE the gain
versus a stronger reference. The single-thread gap to pawnstar is still large (~−258) but narrowing; eval
quality (more/better data, see [[zenith-eval-experiments]]) remains the highest-leverage lever. Next data
scale-ups should keep pushing — 650M was 8 of 16 available PlentyChess shards.
