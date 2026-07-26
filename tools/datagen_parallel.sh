#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonny Reckless
# Fan self-play data generation out across CPU cores. Each worker is an independent ./build/zenith datagen
# process with its own seed, writing one shard. Zenith generates its own dataset (no external data).
#
#   tools/datagen_parallel.sh <gamesPerWorker> <outDir> [workers] [nodes] [openingPlies] [book.epd]
set -euo pipefail

gamesPerWorker="${1:?games per worker}"
outDir="${2:?output directory}"
workers="${3:-$(nproc)}"
nodes="${4:-5000}"
openingPlies="${5:-8}"
book="${6:-}"
net="${7:-}"
engine="${ENGINE:-./build/zenith}"

mkdir -p "$outDir"
echo "datagen: $workers workers x $gamesPerWorker games, nodes=$nodes -> $outDir"

pids=()
for worker in $(seq 1 "$workers"); do
    seed=$((worker * 2654435761))
    "$engine" datagen "$gamesPerWorker" "$outDir/shard_$(printf '%02d' "$worker").txt" \
        "$seed" "$nodes" "$openingPlies" "$book" "$net" 2>"$outDir/shard_$(printf '%02d' "$worker").log" &
    pids+=($!)
done

for pid in "${pids[@]}"; do wait "$pid"; done

positions=$(cat "$outDir"/shard_*.txt | wc -l)
echo "datagen complete: $positions positions across $workers shards in $outDir"
