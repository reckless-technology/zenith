---
name: windows-bench-speed
description: "Zenith is not slower on Windows: measured fairly, bench and datagen match the Linux figures; the gap was E-cores, the 1024 net and x86-64-v2"
metadata:
  type: project
---

**Question (2026-10-18):** `bench` showed 0.9 Mnps and datagen 85-165 pos/s per worker on LEGION5I (i9-13900HX,
Windows 11, MSYS2 CLANG64 clang 23) against 2.15-2.47 Mnps and ~700 pos/s in these notes from Linux. **Answer: Windows
is not slower.** Three things made it look so, none of them the OS:

1. **E-cores.** With other agents' builds filling the P-cores, Windows puts the engine on an E-core: bench pinned to
   E-core 20 runs 0.78 Mnps, a P-core 1.95-2.0. The 0.9 Mnps was an E-core (or a P-core sharing with its busy
   hyperthread sibling: pinned P-core runs dropped to 0.35 Mnps while an MSVC build held the box at 100 %).
2. **Different nets.** The 2.15-2.65M Linux figures predate cap1 (2026-07-29): they are the 512-hidden kb3 era, and the
   1024-hidden net doubles the accumulator. The ~700 pos/s datagen figure is older still, the HCE era (no net).
3. **x86-64-v2 has no AVX2**, and the NNUE's SIMD is `#if defined(__AVX2__)` only: the v2 engine (CI's and the
   portable release's ARCH) benches 1.30 Mnps against 1.95 for native or x86-64-v3. Also `make ARCH=...` used to keep
   a stale build/zenith of another ARCH (no flag dependency) - fixed with build/flags.txt, PR round-b-fixes.

**Like-for-like on the same laptop:** commit bd8deea (C++, kb3 net, the build the Linux "2.47M nps, depth 20 in 2.7 s"
came from; builds in CLANG64 with `-include cstdlib -include algorithm`), UCI `go depth 20` from startpos, Threads 1,
Hash 64: best **2.35M nps, depth 20 in 2.9 s** pinned to a P-core - within 5 % of Linux, on the Balanced power plan.
Old datagen (HCE, 5000 nodes) on a P-core: **734 pos/s** - the Linux ~700. Current datagen (cap2): 318 pos/s on a
P-core, 105 on an E-core, so 85-165 per worker was E-cores and contention.

**Current figures (cap2, bench depth 13, signature 1,390,606):** native and x86-64-v3 best 1.99 Mnps pinned to a
quiet P-core (median 1.95); unpinned the same when the box is quiet; x86-64-v2 1.30-1.37. `go depth 20` from
startpos: 4,781,569 nodes in 2.57 s, 1.86M nps. Ruled out as Windows costs: the stack probes (`___chkstk_ms`, called
by negamax and qsearch since a Position is over 4 KB) - a build without them, on a committed 16 MB stack, benched the
same 2.00 Mnps; winpthreads, malloc and the timer do not enter bench (single thread, inline, the TT allocated once).

**How to measure on this laptop:** check `Get-Counter '\Processor(_Total)\% Processor Time'` and the busiest
processes first; pin with `start "" /affinity 4000 /b /wait build\zenith.exe bench` (logical CPUs 0-15 are the eight
P-cores' threads, 16-31 the E-cores; pick one whose sibling is idle - 12 and 14 were steadiest); take the best and the
median of ten; a run at 0.3-0.8 Mnps means the box is busy, not a regression. Related: [[search-speed-levers]]
(its measurement caveat holds on Windows too).
