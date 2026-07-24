---
name: pawnstar-gap-benchmarks
description: Zenith-vs-pawnstar Elo gaps at various configs (the reference-opponent benchmark)
metadata: 
  node_type: memory
  type: project
  originSessionId: 221925ef-3c7d-4275-96ac-0cd7362cb511
---

Zenith's strength is benchmarked against pawnstar C++ (the reference opponent; user wants Zenith stronger). Gaps measured with fastchess, tc=8+0.08, plentychess openings.

**GOAL FULLY ACHIEVED (2026-07-23): Zenith beats pawnstar at BOTH single thread (+77.7 ± 25.2) and 8 threads
(+107.5 ± 37.8).**

**CRITICAL MEASUREMENT LESSON — pawnstar's UCI defaults are `Threads=32` and `OwnBook=true`.** Every earlier
"single-thread" gap number (−278, −258, −140, −134) never overrode them, so those matches were actually Zenith
1-thread vs pawnstar 32-threads(+book) — a measurement artifact, NOT a real deficit. Always pass explicit
`option.Threads=1 option.OwnBook=false` to pawnstar. The corrected single-thread number (2026-07-23, conc 8):
**Zenith +77.7 ± 25.2 AHEAD.** Supporting data: single-thread nps parity (zenith 2.47M vs pawnstar 2.4–2.5M,
kb3 loaded) and zenith reaches depth 20 in 2.7s where pawnstar reaches 17 in 3.1s.

**IMPORTANT — always match `-concurrency` when comparing gap numbers.** Concurrency changes the timed result a lot: pc2 vs pawnstar was −238±52 at conc 30 but −277±44 at conc 8 (same net, same TC). Cross-run comparisons at different concurrency are meaningless.

Single-thread, matched conditions (conc 8):
- **pc2 (512, 190M plentychess):** −277.5 ± 43.6 vs pawnstar (2026-07-21).
- **kb2 (king buckets, 650M plentychess):** −257.6 ± 40.7 → kb2 closes ~+20 Elo (eval gain, partial transfer).
- **kb2 + correction history + legality oracle:** **−140.4 ± 34.1** (2026-07-22) → full stack closes ~+137 Elo.
- **+ kb3 + inline EP/castle + SPSA-tuned search (current ./zenith):** **−133.6 ± 32.8** (2026-07-23). The single-
  thread gap is roughly HALVED from −278. Note: kb3 (+8) and SPSA (+26.5) are big SELF-PLAY gains but only nudged
  the pawnstar point estimate (−140→−134, within the ±33 noise) — partial transfer again, and hard to resolve at
  ~300 games. The legality-oracle SPEED win (~1:1 transfer) did most of the −278→−140 closure.
- **kb2 vs pc2 self-play: +63.2 ± 29.7** (consistent with the +57 SPRT that shipped kb2).

Session Elo (self-play SPRT): kb2 +57, correction history +10, legality oracle +66 (~+133 cumulative). The
legality-oracle speed win transferred to pawnstar near 1:1 (speed helps vs any opponent), unlike eval gains
(~1/3 transfer) — which is why the full-stack gap closure (~+137) exceeds kb2's eval-only +20.

8 threads each (Lazy SMP), same conditions (conc 3, 8+0.08):
- pc2 net (2026-07-20): −42 ± 48 vs pawnstar — near-even.
- **kb3 + corrhist + legality + EP/castle + SPSA-tuned (2026-07-23): +107.5 ± 37.8 → ZENITH IS STRONGER THAN
  PAWNSTAR AT 8 THREADS.** GOAL ACHIEVED at multi-thread. A +149 Elo swing from the session's work. Zenith's
  Lazy SMP scales much better than pawnstar's, so at equal threads the eval+search gains compound into a
  commanding lead — even though single-thread is still −134 (partial transfer of eval gains). See [[zenith-goal-independence]].

**Key lesson — self-play gains only partially transfer to a much stronger opponent.** kb2 is +63 vs pc2 in
self-play but only ~+20 better against pawnstar (~1/3 transfer). SPRT self-play deltas OVERSTATE the gain
versus a stronger reference. The single-thread gap to pawnstar is still large (~−258) but narrowing; eval
quality (more/better data, see [[zenith-eval-experiments]]) remains the highest-leverage lever. Next data
scale-ups should keep pushing — 650M was 8 of 16 available PlentyChess shards.
