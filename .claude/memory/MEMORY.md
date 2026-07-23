# Memory index

One line per memory. Full content lives in the linked file.

- [prefer-full-names](prefer-full-names.md) — spell out names, avoid terse abbreviations
- [clang-format-consistency](clang-format-consistency.md) — always clang-format all C++ with the repo .clang-format
- [zenith-goal-independence](zenith-goal-independence.md) — beat pawnstar C++ while staying fully independent (own trainer/data/arch/net)
- [pawnstar-reference-opponent](pawnstar-reference-opponent.md) — where the opponent, SPRT harness, GPU, and tooling live
- [training-setup](training-setup.md) — venv/torch, datagen command, GitHub remote + gh CI
- [pgrep-wait-loop-self-match](pgrep-wait-loop-self-match.md) — `pgrep -f` wait loops deadlock by matching themselves; use `pgrep -x`
- [zenith-nnue-pilot-status](zenith-nnue-pilot-status.md) — pipeline validated; pilot net loses to HCE on eval noise (not a bug)
- [pawnstar-gap-benchmarks](pawnstar-gap-benchmarks.md) — Elo gaps vs pawnstar: −238 single-thread, −42 at 8v8; deficit is eval quality
- [zenith-eval-experiments](zenith-eval-experiments.md) — eval-arch experiment log; output buckets NEUTRAL (val loss ≠ Elo), king buckets next
- [search-speed-levers](search-speed-levers.md) — copy-make + 2KB accumulator is the dominant per-node cost; prune-before-make = +66 Elo
- [spsa-tuning](spsa-tuning.md) — SPSA search-param tuner (tools/spsa.py); first pass +26.5 Elo; tune fast tc, validate at 8+0.08
