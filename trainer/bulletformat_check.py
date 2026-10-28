# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonny Reckless
"""The correctness gate for `bulletformat.py`, and its throughput measurement.

Two things are checked, and the first matters more than the second.

**The features must be identical to the ones the text path produces.** A wrong feature index would train a
net on subtly wrong inputs and nothing downstream would notice: the net would load, pass every CRC, and
play badly for no stated reason. So every record is also rendered back to a FEN and put through
`features.position_features`, the route `train._featurise_lines` uses, and the two index sets are compared
element for element. Order within a position is not compared because it is not meaningful -- the model
sums the active features with an EmbeddingBag -- so both sides are sorted first; the piece count, the
padding and every index value are compared exactly.

**The record layout must hold up against chess.** A misread field offset or nibble order can still produce
indices that agree with themselves, so the decoded board is checked for exactly one king per side, for
`ksq` and `opp_ksq` naming those kings' squares, and for a piece count that fits a legal position.

Run it against any bulletformat file:

    python trainer/bulletformat_check.py <records.data> [--positions 5000] [--benchmark]

numpy only; torch is not needed.
"""

import argparse
import sys
import time

import numpy as np

import bulletformat
from features import MAX_ACTIVE_FEATURES, PADDING_INDEX, position_features

PIECE_TYPE_KING = 5


def check_record_layout(records):
    """Decode each record against chess's own invariants. Returns the number of records that failed."""
    failures = 0
    for index in range(records.shape[0]):
        record = records[index]
        occupancy = int(record["occupancy"])
        packed = np.asarray(record["pieces"])
        own_king_squares, opponent_king_squares = [], []
        slot = 0
        working = occupancy
        while working:
            square = (working & -working).bit_length() - 1
            working &= working - 1
            nibble = int(packed[slot // 2] >> (4 * (slot & 1))) & 0x0F
            if (nibble & 7) == PIECE_TYPE_KING:
                (own_king_squares if (nibble >> 3) == 0 else opponent_king_squares).append(square)
            slot += 1

        problems = []
        if len(own_king_squares) != 1 or len(opponent_king_squares) != 1:
            problems.append(f"{len(own_king_squares)} own kings, {len(opponent_king_squares)} opponent kings")
        else:
            if own_king_squares[0] != int(record["own_king_square"]):
                problems.append(f"ksq says {int(record['own_king_square'])}, the king is on {own_king_squares[0]}")
            # opp_ksq is stored already XOR 56, in the opponent's own frame.
            if (opponent_king_squares[0] ^ 56) != int(record["opponent_king_square"]):
                problems.append(f"opp_ksq says {int(record['opponent_king_square'])}, "
                                f"the king is on {opponent_king_squares[0]} ({opponent_king_squares[0] ^ 56} flipped)")
        if bin(occupancy).count("1") > MAX_ACTIVE_FEATURES:
            problems.append(f"{bin(occupancy).count('1')} pieces")
        if int(record["result"]) > 2:
            problems.append(f"result is {int(record['result'])}, not 0, 1 or 2")
        if np.any(np.asarray(record["extra"]) != 0):
            problems.append("the extra bytes are not zero")
        if problems:
            failures += 1
            if failures <= 5:
                print(f"  record {index}: {'; '.join(problems)}")
    return failures


def check_features_match(records):
    """Compare the vectorised features against `features.position_features`. Returns the mismatch count."""
    own_indices, opponent_indices, _, _ = bulletformat.featurise_records(records)
    mismatches = 0
    for index in range(records.shape[0]):
        fen = bulletformat.record_to_fen(records[index])
        _, text_own, text_opponent = position_features(fen)
        vectorised_own = own_indices[index][own_indices[index] != PADDING_INDEX]
        vectorised_opponent = opponent_indices[index][opponent_indices[index] != PADDING_INDEX]
        matches = (np.array_equal(np.sort(vectorised_own), np.sort(np.asarray(text_own, dtype=np.int16)))
                   and np.array_equal(np.sort(vectorised_opponent),
                                      np.sort(np.asarray(text_opponent, dtype=np.int16))))
        if not matches:
            mismatches += 1
            if mismatches <= 3:
                print(f"  record {index} differs\n    {fen}"
                      f"\n    vectorised own {sorted(vectorised_own.tolist())}"
                      f"\n    text       own {sorted(text_own)}"
                      f"\n    vectorised opp {sorted(vectorised_opponent.tolist())}"
                      f"\n    text       opp {sorted(text_opponent)}")
    return mismatches


def benchmark(path):
    """Featurisation throughput, single-threaded, warm and then streaming off the file."""
    print("\n=== throughput, one thread ===")
    for exponent in (16, 18, 20):
        size = 1 << exponent
        block = np.array(bulletformat.read_records(path, count=size))
        best = 0.0
        for _ in range(5):
            start = time.perf_counter()
            bulletformat.featurise_records(block)
            best = max(best, size / (time.perf_counter() - start))
        print(f"  {size:>9,} records already in RAM: {best / 1e6:.2f}M positions/sec")

    total = min(4 << 20, bulletformat.record_count_of(path))
    block_size = 1 << 18
    done = 0
    start = time.perf_counter()
    while done < total:
        size = min(block_size, total - done)
        bulletformat.featurise_records(bulletformat.read_records(path, count=size, first_record=done))
        done += size
    elapsed = time.perf_counter() - start
    print(f"  {total:,} records streamed off the file: {total / elapsed / 1e6:.2f}M positions/sec")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("records", help="a bulletformat file (32-byte records, no header)")
    parser.add_argument("--positions", type=int, default=5000,
                        help="how many records to check (default 5000)")
    parser.add_argument("--benchmark", action="store_true", help="also measure featurisation throughput")
    arguments = parser.parse_args()

    print(f"{arguments.records}: {bulletformat.record_count_of(arguments.records):,} records")
    records = bulletformat.read_records(arguments.records, count=arguments.positions)
    print(f"\n=== record layout against chess's invariants, {records.shape[0]:,} records ===")
    layout_failures = check_record_layout(records)
    print(f"  {layout_failures} failed")

    print(f"\n=== features vs features.position_features, {records.shape[0]:,} positions ===")
    feature_mismatches = check_features_match(records)
    print(f"  {feature_mismatches} mismatched")

    if arguments.benchmark:
        benchmark(arguments.records)

    good = layout_failures == 0 and feature_mismatches == 0
    print("\n[PASS] bulletformat featurisation matches the text path exactly"
          if good else "\n[FAIL] see above")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
