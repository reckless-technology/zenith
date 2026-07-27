# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonny Reckless
"""Verification gate: the C engine's integer NNUE eval must equal the Python reference byte-for-byte.

Loads a .nnue file, evaluates a set of FENs with `features.integer_eval` (the reference), runs the same
FENs through `./build/zenith nnueeval <net>`, and compares. Any nonzero difference means the engine loader or
forward pass diverges from the trainer's export contract.

    python trainer/verify.py --net nets/zenith-pilot.nnue --fens data/pilot/shard_01.txt --count 2000
"""

import argparse
import struct
import subprocess

import numpy as np

from features import (HIDDEN_SIZE, INPUT_FEATURES, NNUE_MAGIC, NNUE_MAGIC_V3, NUM_OUTPUT_BUCKETS,
                      integer_eval, position_features)


def load_net(path):
    """Load a v4 net; a v3 net's single output head is broadcast to all buckets (identical evals)."""
    with open(path, "rb") as handle:
        magic = handle.read(8)
        if magic not in (NNUE_MAGIC, NNUE_MAGIC_V3):
            raise SystemExit(f"bad magic in {path}: {magic!r}")
        transformer = np.frombuffer(handle.read(INPUT_FEATURES * HIDDEN_SIZE * 2), dtype=np.int16)
        transformer = transformer.reshape(INPUT_FEATURES, HIDDEN_SIZE).astype(np.int64)
        transformer_bias = np.frombuffer(handle.read(HIDDEN_SIZE * 2), dtype=np.int16).astype(np.int64)
        if magic == NNUE_MAGIC_V3:
            one = np.frombuffer(handle.read(2 * HIDDEN_SIZE * 2), dtype=np.int16).astype(np.int64)
            output_weight = np.tile(one, (NUM_OUTPUT_BUCKETS, 1))
            output_bias = np.full(NUM_OUTPUT_BUCKETS, struct.unpack("<i", handle.read(4))[0], dtype=np.int64)
        else:
            output_weight = np.frombuffer(handle.read(NUM_OUTPUT_BUCKETS * 2 * HIDDEN_SIZE * 2), dtype=np.int16)
            output_weight = output_weight.reshape(NUM_OUTPUT_BUCKETS, 2 * HIDDEN_SIZE).astype(np.int64)
            output_bias = np.frombuffer(handle.read(NUM_OUTPUT_BUCKETS * 4), dtype=np.int32).astype(np.int64)
    return transformer, transformer_bias, output_weight, output_bias


def collect_fens(path, count):
    fens = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            fens.append(line.split(";")[0])
            if len(fens) >= count:
                break
    return fens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", required=True)
    parser.add_argument("--fens", required=True, help="a datagen shard (or any file with FEN as first ;-field)")
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--engine", default="./build/zenith")
    args = parser.parse_args()

    transformer, transformer_bias, output_weight, output_bias = load_net(args.net)
    fens = collect_fens(args.fens, args.count)

    reference = []
    for fen in fens:
        _, own, opponent = position_features(fen)
        reference.append(integer_eval(transformer, transformer_bias, output_weight, output_bias, own, opponent))

    completed = subprocess.run([args.engine, "nnueeval", args.net], input="\n".join(fens) + "\n",
                               capture_output=True, text=True, check=True)
    engine = [int(x) for x in completed.stdout.split()]

    if len(engine) != len(reference):
        raise SystemExit(f"count mismatch: engine {len(engine)} vs reference {len(reference)}")

    differences = np.array(engine) - np.array(reference)
    max_absolute = int(np.max(np.abs(differences))) if len(differences) else 0
    mismatches = int(np.sum(differences != 0))
    print(f"checked {len(fens)} positions  max|engine-reference| = {max_absolute} cp  mismatches = {mismatches}")
    if max_absolute == 0:
        print("PASS: engine integer eval is bit-identical to the trainer reference")
    else:
        worst = int(np.argmax(np.abs(differences)))
        print(f"FAIL: e.g. {fens[worst]}  engine={engine[worst]} reference={reference[worst]}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
