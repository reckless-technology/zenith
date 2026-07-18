#!/usr/bin/env bash
# Self-play SPRT: a candidate Zenith build vs a baseline build (measures the Elo of a change).
# Usage: tools/sprt.sh [cand] [base] [rounds]   Env: TC (default 8+0.08), ELO0/ELO1 (default 0/5), CONCURRENCY.
set -euo pipefail
CAND="${1:-./zenith}"; BASE="${2:-./zenith-base}"; ROUNDS="${3:-4000}"
TC="${TC:-8+0.08}"; ELO0="${ELO0:-0}"; ELO1="${ELO1:-5}"
OPEN="${OPENINGS:-$HOME/pawnstar_nnue/openings.epd}"; CC="${CONCURRENCY:-5}"
for f in "$CAND" "$BASE"; do [ -x "$f" ] || { echo "missing binary: $f" >&2; exit 1; }; done
echo "SPRT cand=$CAND base=$BASE TC=$TC elo[$ELO0,$ELO1] rounds=$ROUNDS"
exec cutechess-cli \
  -engine cmd="$CAND" name=cand -engine cmd="$BASE" name=base \
  -each proto=uci tc="$TC" -rounds "$ROUNDS" -games 2 -repeat \
  -openings file="$OPEN" format=epd order=random -draw movenumber=40 movecount=8 score=20 \
  -resign movecount=3 score=600 -sprt elo0="$ELO0" elo1="$ELO1" alpha=0.05 beta=0.05 \
  -concurrency "$CC" -ratinginterval 20
