#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonny Reckless
"""Generate a synthetic net in the shared `RNNUE1` container for CI: deterministic pseudo-random weights.

The NNUE correctness gates are weight-agnostic — nnuecheck verifies incremental accumulator maintenance
against a full refresh (identical for any weights), and the loader and forward pass only need a
well-formed file — so CI can exercise the whole NNUE path (loader, accumulator updates, finny refresh
cache, integer forward, UCI EvalFile) without committing a real 12.6MB network. Weights are kept small
(|w| <= 50) so accumulator sums stay far from int16 range regardless of position.

The shape and the container both come from `trainer/rnnue1.py`, which is stdlib-only for exactly this
reason: CI runs this script on a machine with no trainer environment.

Usage: python3 tools/make_test_net.py <out.nnue>
"""
import os
import random
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "trainer"))
import rnnue1  # noqa: E402  (the path above is what makes this importable)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: make_test_net.py <out.nnue>")
    rng = random.Random(0x5EED)
    sections = []
    # feature-transformer weights, feature-major, then the feature-transformer bias
    for _ in range(rnnue1.INPUT_FEATURES):
        sections.append(struct.pack(f"<{rnnue1.HIDDEN_SIZE}h",
                                    *(rng.randint(-50, 50) for _ in range(rnnue1.HIDDEN_SIZE))))
    sections.append(struct.pack(f"<{rnnue1.HIDDEN_SIZE}h",
                                *(rng.randint(-50, 50) for _ in range(rnnue1.HIDDEN_SIZE))))
    # output weights per material bucket (own then opponent half), then one int32 bias per bucket
    for _ in range(rnnue1.NUM_OUTPUT_BUCKETS):
        sections.append(struct.pack(f"<{2 * rnnue1.HIDDEN_SIZE}h",
                                    *(rng.randint(-50, 50) for _ in range(2 * rnnue1.HIDDEN_SIZE))))
    sections.append(struct.pack(f"<{rnnue1.NUM_OUTPUT_BUCKETS}i",
                                *(rng.randint(-1000, 1000) for _ in range(rnnue1.NUM_OUTPUT_BUCKETS))))
    written = rnnue1.write_net(sys.argv[1], b"".join(sections))
    print(f"wrote {sys.argv[1]} ({written} bytes)")


if __name__ == "__main__":
    main()
