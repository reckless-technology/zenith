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
#         TC (default 8+0.08), ELO0/ELO1 (default 0/5), CONCURRENCY, OPENINGS.
set -euo pipefail

CAND="${CAND:-./build/zenith}"
BASE="${BASE:-./build/zenith-base}"
ROUNDS="${1:-2000}"
TC="${TC:-8+0.08}"
ELO0="${ELO0:-0}"
ELO1="${ELO1:-5}"
OPENINGS="${OPENINGS:-$HOME/pawnstar_nnue/openings.epd}"
CONCURRENCY="${CONCURRENCY:-$(( $(nproc) - 2 ))}"

FASTCHESS="$(command -v fastchess || true)"
[ -z "$FASTCHESS" ] && [ -x "$HOME/pawnstar_nnue/fastchess/fastchess" ] && FASTCHESS="$HOME/pawnstar_nnue/fastchess/fastchess"
[ -z "$FASTCHESS" ] && { echo "fastchess not found (PATH or ~/pawnstar_nnue/fastchess/fastchess)" >&2; exit 1; }

for engine in "$CAND" "$BASE"; do
    command -v "$engine" >/dev/null 2>&1 || [ -x "$engine" ] || { echo "missing engine: $engine" >&2; exit 1; }
done
[ -f "$OPENINGS" ] || { echo "missing openings: $OPENINGS" >&2; exit 1; }

# Optional per-engine EvalFile (NNUE net). Absent => that side uses its default (zenith: the embedded net).
cand_net_arg=(); [ -n "${CAND_NET:-}" ] && cand_net_arg=(option.EvalFile="$CAND_NET")
base_net_arg=(); [ -n "${BASE_NET:-}" ] && base_net_arg=(option.EvalFile="$BASE_NET")
# Optional per-engine working directory (e.g. pawnstar loads its net via a relative path).
cand_dir_arg=(); [ -n "${CAND_DIR:-}" ] && cand_dir_arg=(dir="$CAND_DIR")
base_dir_arg=(); [ -n "${BASE_DIR:-}" ] && base_dir_arg=(dir="$BASE_DIR")

echo "SPRT  cand=$CAND ${CAND_NET:+net=$CAND_NET}  base=$BASE ${BASE_NET:+net=$BASE_NET}  TC=$TC elo[$ELO0,$ELO1] rounds=$ROUNDS"
exec "$FASTCHESS" \
    -engine cmd="$CAND" name=cand "${cand_net_arg[@]}" "${cand_dir_arg[@]}" \
    -engine cmd="$BASE" name=base "${base_net_arg[@]}" "${base_dir_arg[@]}" \
    -each proto=uci tc="$TC" \
    -rounds "$ROUNDS" -games 2 -repeat \
    -openings file="$OPENINGS" format=epd order=random \
    -draw movenumber=40 movecount=8 score=20 -resign movecount=3 score=600 \
    -sprt elo0="$ELO0" elo1="$ELO1" alpha=0.05 beta=0.05 \
    -concurrency "$CONCURRENCY" -ratinginterval 20
