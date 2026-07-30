---
name: zenith-opening-book
description: The self-play-built Polyglot opening book — recipe, round results, and lessons
metadata:
  type: project
---

Zenith's opening book (`books/zenith-book-r3.bin`, committed; 321KB, ~9,900 positions, ~20k weighted
moves) is built ENTIRELY from Zenith self-play by `tools/build_book.py` — no external book or game data
(consistent with [[zenith-goal-independence]]).

**Recipe (validated over three gated rounds):** grow a position DAG from startpos best-first by reach
probability; per position: screen every legal move (150k nodes) -> rescore survivors (2M nodes, keep <=4
within 25cp) -> self-play playout tiebreaks when the top moves sit within 15cp (3 color-swapped game
pairs, node-budget jitter for variety; ~10-15% of moves) -> negamax backup over the DAG. Emit key-sorted
big-endian Polyglot (castling = king-takes-rook; weight ~ 100 - 2*cp_gap). All chess knowledge comes from
engine subcommands (`moves`/`polykey`/`applymoves`/`san`), so builder and book.c cannot drift.

**Results (book vs bookless, 8+0.08, ob2-era 512 engine):** 97 positions +17+/-17; 943 +40+/-12; 9,910
+28+/-11 — the bookless gate PLATEAUS (~+30) because deep positions are reached too rarely to move it.
But head-to-head r3 vs r2 (both booked): **+23 +/- 11 for r3** — depth keeps paying in book-vs-book play
even when the bookless gate saturates. Judge books head-to-head, not only vs bookless.

**Lessons:**
- Tight keep-margins matter: 40cp let junk sidelines (h6/a6) into round 1; 25cp + playouts fixed it.
- The book only contains moves the ENGINE rates near-best (zenith books no 1.d4 — it scores it 29cp below
  e4). Fine for self-play; a general tournament book needs ASYMMETRIC margins (wide for opponent moves,
  tight for own) so the book has replies to lines zenith wouldn't play (e.g. as Black vs 1.d4). Polyglot
  needs no format support for this: replies live under the position-after-the-move key. UNTESTED extension.
- book.c had a latent mod-by-zero when a position's matched entries were all weight-0 (fixed with the ship).
- Long-running pipelines must own a DEDICATED binary copy (data/book/zenith-worker) — rebuilding
  ./build/zenith under a live consumer crashes it (exec during the non-executable write window).
- Store is resumable JSON keyed by polyglot key (data/book/store.json); tools/show_book.py displays any
  Polyglot book as SAN lines.

**Round 4 — coverage book (zenith-book-r4.bin, 487KB, ~14k positions), SHIPPED 2026-07-30.** Adds
opponent coverage: moves within 90cp/top-4 at plies <10 expand (never emitted as our pick unless backup
promotes them within the 25cp playable margin — which it DID for 1.c4/1.d4 at the root: fresh deep
rescoring put them 18cp off best, so the book now opens them ~20% each). Gates: r4-vs-r3 self-play equal
(50.9%); vs pawnstar all three configs (r4/r3/bookless) within noise at 68-70% — **at a +135 Elo skill
gap, book effects are below the measurement floor** (800 games/config, +/-17 Elo). Coverage value is
structural, not Elo-provable with available opponents. Ship rationale: strict superset, zero measured
cost. Lesson: book gates need a NEAR-PEER foreign opponent to resolve anything; self-play gates cannot
see coverage by construction.
