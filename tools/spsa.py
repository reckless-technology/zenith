#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonny Reckless
"""SPSA tuner for Zenith's search parameters (fastchess has no built-in SPSA).

Each iteration perturbs all parameters simultaneously by +/- c_k*Delta (random signs), plays a small
self-play match between the theta+ and theta- engines (same binary, different UCI option values), and nudges
theta toward whichever side scored better (Simultaneous Perturbation Stochastic Approximation, Spall).
Standard gain schedule: c_k = c0/k^0.101, a_k = a0/(k+A)^0.602. Checkpoints theta to a file each iteration.

    python tools/spsa.py --engine ./build/zenith-spsa --net nets/zenith-kb3.nnue \
        --openings ~/pawnstar_nnue/openings.epd --fastchess ~/pawnstar_nnue/fastchess/fastchess \
        --iters 800 --games 8 --concurrency 8 --tc 8+0.08 --out tools/spsa_state.json
"""
import argparse
import json
import os
import random
import re
import subprocess
import time

# (name, initial, min, max) — must match the UCI spin options exposed by the engine. Initials mirror the
# CURRENT SearchParams defaults in search.c (the first SPSA pass, already baked in), so a fresh run continues
# from the shipped engine rather than re-tuning from the pre-pass values.
PARAMS = [
    ("RfpMargin", 56, 20, 200),
    ("NmpDivisor", 199, 50, 600),
    ("LmpBase", 5, 1, 10),
    ("FutilityBase", 94, 0, 300),
    ("FutilityMargin", 98, 30, 200),
    ("SeeCaptureMargin", 97, 20, 300),
    ("LmrBase", 90, 0, 200),
    ("LmrDivisor", 220, 100, 400),
    ("SingularMargin", 3, 1, 8),
    ("AspirationDelta", 21, 5, 60),
    ("HistoryMax", 402, 100, 1200),
]

GAMMA, ALPHA = 0.101, 0.602


def clip(value, low, high):
    return max(low, min(high, value))


def run_match(args, plus, minus):
    """Play args.games games between the theta+ and theta- option sets; return theta+'s points minus
    theta-'s points (draws cancel), i.e. (wins - losses) from theta+'s perspective."""
    def opts(values):
        out = [f"option.EvalFile={args.net}"]
        for (name, _, _, _), value in zip(PARAMS, values):
            out.append(f"option.{name}={int(round(value))}")
        return out

    rounds = max(1, args.games // 2)
    cmd = [args.fastchess,
           "-engine", f"cmd={args.engine}", "name=plus", *opts(plus),
           "-engine", f"cmd={args.engine}", "name=minus", *opts(minus),
           "-each", "proto=uci", f"tc={args.tc}",
           "-rounds", str(rounds), "-games", "2", "-repeat",
           "-openings", f"file={args.openings}", "format=epd", "order=random",
           "-draw", "movenumber=40", "movecount=8", "score=20", "-resign", "movecount=3", "score=600",
           "-concurrency", str(args.concurrency)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # fastchess prints "Games: N, Wins: W, Losses: L, Draws: D, ..." with W/L from the first engine (plus).
    # Take the last such line (final tally).
    matches = re.findall(r"Games:\s*\d+,\s*Wins:\s*(\d+),\s*Losses:\s*(\d+),\s*Draws:\s*(\d+)", result.stdout)
    if not matches:
        return 0, 0
    wins, losses, draws = (int(x) for x in matches[-1])
    return (wins - losses), (wins + losses + draws)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--net", required=True)
    parser.add_argument("--openings", required=True)
    parser.add_argument("--fastchess", required=True)
    parser.add_argument("--iters", type=int, default=800)
    parser.add_argument("--games", type=int, default=8, help="games per iteration (both option sets)")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--tc", default="8+0.08")
    parser.add_argument("--out", default="tools/spsa_state.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", default=None,
                        help="state json from a previous round: continue its theta AND its iteration count, "
                             "so chained rounds form one gradient-descent trajectory (the gain schedule keeps "
                             "decaying instead of restarting)")
    args = parser.parse_args()
    random.seed(args.seed)

    theta = [float(init) for (_, init, _, _) in PARAMS]
    start_iteration = 1
    if args.resume:
        with open(args.resume) as handle:
            prior = json.load(handle)
        theta = [float(prior["theta"][name]) for (name, _, _, _) in PARAMS]
        start_iteration = prior["iteration"] + 1
        print(f"resuming from {args.resume}: iteration {start_iteration}, theta "
              + "  ".join(f"{k}={v}" for k, v in prior["theta"].items()), flush=True)

    # Per-param perturbation c0 (~8% of range, >=1) and learning rate a0. a0 is scaled so an early full-signal
    # iteration nudges theta by ~a0/(2c0); we pick a0 to move a few % of range per confident iteration.
    c0 = [max(1.0, 0.08 * (hi - lo)) for (_, _, lo, hi) in PARAMS]
    a0 = [0.15 * (hi - lo) for (_, _, lo, hi) in PARAMS]
    A = max(20, args.iters // 10)

    start = time.time()
    for iteration in range(start_iteration, args.iters + 1):
        ck = [c / iteration ** GAMMA for c in c0]
        ak = [a / (iteration + A) ** ALPHA for a in a0]
        delta = [random.choice((-1, 1)) for _ in PARAMS]
        plus, minus = [], []
        for i, (_, _, lo, hi) in enumerate(PARAMS):
            plus.append(clip(theta[i] + ck[i] * delta[i], lo, hi))
            minus.append(clip(theta[i] - ck[i] * delta[i], lo, hi))
        margin, played = run_match(args, plus, minus)
        if played:
            reward = margin / played  # in [-1, 1], theta+'s edge
            for i, (_, _, lo, hi) in enumerate(PARAMS):
                # SPSA update: gradient ~ reward*delta/(2ck); step a_k * that.
                theta[i] = clip(theta[i] + ak[i] * reward * delta[i] / (2 * ck[i]) * c0[i], lo, hi)
        rounded = {name: int(round(theta[i])) for i, (name, _, _, _) in enumerate(PARAMS)}
        state = {"iteration": iteration, "iters": args.iters, "elapsed_s": round(time.time() - start),
                 "theta": rounded, "last_margin": margin, "last_games": played}
        with open(args.out, "w") as handle:
            json.dump(state, handle, indent=2)
        if iteration % 10 == 0 or iteration == 1:
            print(f"iter {iteration:4d}/{args.iters}  margin {margin:+d}/{played}  "
                  f"{time.time() - start:.0f}s  " + "  ".join(f"{k}={v}" for k, v in rounded.items()), flush=True)

    print("SPSA DONE. final:", json.dumps({name: int(round(theta[i])) for i, (name, _, _, _) in enumerate(PARAMS)}))


if __name__ == "__main__":
    main()
