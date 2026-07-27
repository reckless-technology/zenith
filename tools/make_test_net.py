#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonny Reckless
"""Generate a synthetic ZNNUE4 net for CI: valid format, deterministic pseudo-random weights.

The NNUE correctness gates are weight-agnostic — nnuecheck verifies incremental accumulator maintenance
against a full refresh (identical for any weights), and the loader/forward only need a well-formed file —
so CI can exercise the whole NNUE path (loader, accumulator updates, finny refresh cache, integer forward,
UCI EvalFile) without committing a real 6.3MB network. Weights are kept small (|w| <= 50) so accumulator
sums stay far from int16 range regardless of position.

Usage: python3 tools/make_test_net.py <out.nnue>
"""
import random
import struct
import sys

HIDDEN = 512
OUTPUT_BUCKETS = 8
INPUT_FEATURES = 8 * 768  # king buckets x per-bucket features; must match nnue.c


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: make_test_net.py <out.nnue>")
    rng = random.Random(0x5EED)
    with open(sys.argv[1], "wb") as out:
        out.write(b"ZNNUE4\x00\x00")
        # feature-transformer weights, feature-major
        for _ in range(INPUT_FEATURES):
            out.write(struct.pack(f"<{HIDDEN}h", *(rng.randint(-50, 50) for _ in range(HIDDEN))))
        # feature-transformer bias
        out.write(struct.pack(f"<{HIDDEN}h", *(rng.randint(-50, 50) for _ in range(HIDDEN))))
        # output weights per material bucket (own + opponent halves each)
        for _ in range(OUTPUT_BUCKETS):
            out.write(struct.pack(f"<{2 * HIDDEN}h", *(rng.randint(-50, 50) for _ in range(2 * HIDDEN))))
        # output bias per bucket
        out.write(struct.pack(f"<{OUTPUT_BUCKETS}i", *(rng.randint(-1000, 1000) for _ in range(OUTPUT_BUCKETS))))
    print(f"wrote {sys.argv[1]}")


if __name__ == "__main__":
    main()
