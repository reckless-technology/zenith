---
name: sprt-progress-reporting
description: "Jonny wants periodic interim progress reports from long SPRT runs, not silence-until-verdict"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 221925ef-3c7d-4275-96ac-0cd7362cb511
  modified: 2026-07-31T06:04:48.722Z
---

Jonny explicitly likes the pawnstar-era practice of SPRT matches reporting progress regularly (games
played, score, Elo estimate, LLR) while running.

**Why:** long SPRTs (hours) that only report at the verdict leave him blind to how things are shaping up;
he asked for status repeatedly during a silent 8-hour validation run and then said he liked the old way.

**How to apply:** never pipe a long fastchess run through a filter that discards interim output. Retain
the FULL fastchess log (`> run.log 2>&1`), then attach a Monitor that emits the periodic
`Games:`/`Elo`/`LLR:` lines every 50 games (Jonny's stated cadence: `Games: [0-9]*[05]0,`) plus
the final `SPRT (` verdict line. Relay each milestone to Jonny in one line. Same pattern for SPSA
(per-100-iteration lines) and training (per-epoch lines) — already standard; SPRT was the gap.

**Engine invariants (Jonny's explicit rule):** every match harness sets BOTH engines explicitly — never
rely on defaults: `option.Threads=1 option.Hash=64 option.OwnBook=false`, and no pondering (Zenith has no
Ponder option; fastchess does not ponder — re-verify if either changes). Defaults drift (the embedded book
made OwnBook semantics richer; pawnstar's Threads=32/OwnBook=true burned a whole measurement once).
Verify the pins landed via /proc/<fastchess>/cmdline, not by trusting the launcher script.

Related hazard when restarting such runs: [[pgrep-wait-loop-self-match]] — kill by PID, never by
`pkill -f` patterns that appear in your own command line.
