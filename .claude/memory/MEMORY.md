# Memory index

One line per memory. Full content lives in the linked file.

- [zenith-goal-independence](zenith-goal-independence.md) — beat pawnstar C++ while staying fully independent (own trainer/data/arch/net)
- [pawnstar-reference-opponent](pawnstar-reference-opponent.md) — where the opponent, SPRT harness, GPU, and tooling live
- [training-setup](training-setup.md) — venv/torch, datagen command, GitHub remote + gh CI
- [zenith-nnue-pilot-status](zenith-nnue-pilot-status.md) — pipeline validated; pilot net loses to HCE on eval noise (not a bug)
- [pawnstar-gap-benchmarks](pawnstar-gap-benchmarks.md) — Elo gaps vs pawnstar: −238 single-thread, −42 at 8v8; deficit is eval quality
- [zenith-eval-experiments](zenith-eval-experiments.md) — experiment log: buckets, data scaling, pawnstar-idea adoption (techniques don't transfer; fix measured costs)
- [search-speed-levers](search-speed-levers.md) — copy-make + 2KB accumulator is the dominant per-node cost; prune-before-make = +66 Elo
- [spsa-tuning](spsa-tuning.md) — SPSA search-param tuner (tools/spsa.py); first pass +26.5 Elo; tune fast tc, validate at 8+0.08
- [setwise-elegance-nonregression-gate](setwise-elegance-nonregression-gate.md) — elegance refactors ship on non-regression SPRT (ELO0=-5 ELO1=0), not the gain test
- [zenith-opening-book](zenith-opening-book.md) — self-play-built Polyglot book: recipe, gates, head-to-head lesson
- [sprt-progress-reporting](sprt-progress-reporting.md) — SPRTs report progress live; pin Threads/Hash/OwnBook explicitly on both engines

- Naming, clang-format (org-wide pin **22.1.8** since 2026-09-26), the `pgrep -f` self-match trap and the
  simple-config preference moved to `reckless-technology/claude-skills`, skill `reckless-working-practice` — none of them were about Zenith