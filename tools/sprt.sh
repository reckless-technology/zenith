#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonny Reckless
# Self-play / cross-engine SPRT via fastchess (cutechess-cli is not installed on this box). Measures the
# Elo of a change: a candidate engine/net vs a baseline. Works for:
#   * new net vs shipped: CAND=./build/zenith CAND_NET=nets/x.nnue  BASE=./build/zenith          (BASE_NET unset -> embedded net)
#   * version vs version: two zenith binaries
#   * vs pawnstar:      BASE=/home/jonny/work/pawnstar/build/pawnstar BASE_NET=.../pawnstar-v12.bin
#
# Usage:  tools/sprt.sh [rounds]
# Env:    CAND (default ./build/zenith), BASE (default ./build/zenith-base), CAND_NET, BASE_NET (EvalFile paths),
#         TC (default 8+0.08), ELO0/ELO1 (default 0/5), CONCURRENCY, HASH, OPENINGS (REQUIRED).
#
# Every engine option a measurement depends on is pinned below rather than left to the UCI default, the
# same way `pawnstar/tools/pawnstar-match.sh` does it. Threads especially: zenith's own default is
# `platform_cpu_count() / 2` (src/uci.c), so an unpinned run on a 16-CPU machine plays 8-thread engines -
# and with CONCURRENCY also sized from nproc, it oversubscribes the box by an order of magnitude and
# reports scheduler noise as Elo. A harness pins Threads=1; src/uci.c's own comment says so.
set -euo pipefail

CAND="${CAND:-./build/zenith}"
BASE="${BASE:-./build/zenith-base}"
ROUNDS="${1:-2000}"
TC="${TC:-8+0.08}"
ELO0="${ELO0:-0}"
ELO1="${ELO1:-5}"
OPENINGS="${OPENINGS:-}"
HASH="${HASH:-16}"
# One game per physical core, less one for fastchess itself, assuming SMT (nproc counts hyperthreads).
# On reckless-ci-lin-0 (16 logical / 8 physical) that is 7, which measured 1.28 M nps per engine against
# ~1.3 M idle (2026-10-30). The previous `nproc - 2` default was 14 there, and a 14-way run in an earlier
# campaign on that machine left each engine at 44-47% of its idle rate - enough to swamp the Elo
# difference the run exists to measure. Set CONCURRENCY explicitly on a machine without SMT.
logical_cpus="$(nproc)"
CONCURRENCY="${CONCURRENCY:-$(( logical_cpus / 2 - 1 > 0 ? logical_cpus / 2 - 1 : 1 ))}"

FASTCHESS="$(command -v fastchess || true)"
[ -z "$FASTCHESS" ] && [ -x "$HOME/pawnstar_nnue/fastchess/fastchess" ] && FASTCHESS="$HOME/pawnstar_nnue/fastchess/fastchess"
[ -z "$FASTCHESS" ] && { echo "fastchess not found (PATH or ~/pawnstar_nnue/fastchess/fastchess)" >&2; exit 1; }

for engine in "$CAND" "$BASE"; do
    command -v "$engine" >/dev/null 2>&1 || [ -x "$engine" ] || { echo "missing engine: $engine" >&2; exit 1; }
done
# No opening set ships in this repo (books/ holds Polyglot .bin books, which fastchess -openings cannot
# read - it wants EPD or PGN), so there is nothing sensible to default to and guessing a sibling repo's
# scratch path only fails later and less clearly.
if [ -z "$OPENINGS" ]; then
    echo "tools/sprt.sh: set OPENINGS to an opening set for fastchess to play from." >&2
    echo "  OPENINGS=/path/to/openings.epd tools/sprt.sh [rounds]" >&2
    echo "  It must be EPD or PGN; this repo ships no opening set, and Polyglot books under books/ are" >&2
    echo "  not one. Without openings every game would start from the same position." >&2
    exit 1
fi
[ -f "$OPENINGS" ] || { echo "missing openings: $OPENINGS" >&2; exit 1; }

# Optional per-engine EvalFile (NNUE net). Absent => that side uses its default (zenith: the embedded net).
cand_net_arg=(); [ -n "${CAND_NET:-}" ] && cand_net_arg=(option.EvalFile="$CAND_NET")
base_net_arg=(); [ -n "${BASE_NET:-}" ] && base_net_arg=(option.EvalFile="$BASE_NET")
# Optional per-engine working directory (e.g. pawnstar loads its net via a relative path).
cand_dir_arg=(); [ -n "${CAND_DIR:-}" ] && cand_dir_arg=(dir="$CAND_DIR")
base_dir_arg=(); [ -n "${BASE_DIR:-}" ] && base_dir_arg=(dir="$BASE_DIR")

echo "SPRT  cand=$CAND ${CAND_NET:+net=$CAND_NET}  base=$BASE ${BASE_NET:+net=$BASE_NET}  TC=$TC elo[$ELO0,$ELO1] rounds=$ROUNDS"
echo "      Threads=1 Hash=${HASH}MB OwnBook=false per engine, concurrency=$CONCURRENCY, openings=$OPENINGS"
exec "$FASTCHESS" \
    -engine cmd="$CAND" name=cand "${cand_net_arg[@]}" "${cand_dir_arg[@]}" \
    -engine cmd="$BASE" name=base "${base_net_arg[@]}" "${base_dir_arg[@]}" \
    -each proto=uci tc="$TC" option.Threads=1 option.Hash="$HASH" option.OwnBook=false \
    -rounds "$ROUNDS" -games 2 -repeat \
    -openings file="$OPENINGS" format=epd order=random \
    -draw movenumber=40 movecount=8 score=20 -resign movecount=3 score=600 \
    -sprt elo0="$ELO0" elo1="$ELO1" alpha=0.05 beta=0.05 \
    -concurrency "$CONCURRENCY" -ratinginterval 20
