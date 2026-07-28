#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonny Reckless
"""Build a Polyglot opening book from scratch by Zenith self-play — no external book or game data.

The builder grows a scored opening DAG from the start position, best-first:

  1. SCREEN every legal move of a frontier position with a cheap fixed-node search.
  2. RESCORE the survivors (within --screen-margin of the best, at most --max-candidates) deeper.
  3. PLAYOUT near-ties: when kept moves sit within --playout-margin of the best, play real
     Zenith-vs-Zenith game pairs from each candidate child (color-swapped, node-budget jitter for
     variety) and blend the outcome into the move weights.
  4. BACK UP values over the DAG (negamax over expanded children), so upstream refutations demote
     whole lines, then EXPAND the highest-priority frontier (reach probability x candidate count).

All chess knowledge lives in the engine binary: legal moves (`zenith moves`), Polyglot keys
(`zenith polykey`, bookcheck-validated), child FENs (`zenith applymoves`), and every score (UCI
searches with an explicit EvalFile). The builder is pure orchestration; the store is a resumable
JSON file. `--emit` writes the .bin (big-endian {key, move, weight, learn}, key-sorted, castling in
Polyglot king-takes-rook form to match book.c's encoder); `--match` plays a book-vs-bookless
fastchess match as the round's Elo gate.

Typical round:
  python3 tools/build_book.py --state data/book/store.json --expand 100 --emit books/zenith-r1.bin
  python3 tools/build_book.py --state data/book/store.json --match books/zenith-r1.bin --games 400
"""
import argparse
import json
import math
import os
import queue
import re
import struct
import subprocess
import threading
import time

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
MATE_CP = 30000  # mate scores mapped into centipawns, safely above any static eval


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---- engine access -------------------------------------------------------------------------------------


class Engine:
    """One persistent Zenith UCI process pinned to a single thread and an explicit net."""

    def __init__(self, binary, net):
        self.process = subprocess.Popen([binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                        bufsize=1)
        self.send("uci")
        self.wait_for("uciok")
        self.send("setoption name Threads value 1")
        if net:
            self.send(f"setoption name EvalFile value {net}")
        self.send("isready")
        self.wait_for("readyok")

    def send(self, line):
        self.process.stdin.write(line + "\n")

    def wait_for(self, token):
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("engine died")
            if line.startswith(token):
                return line

    def search_nodes(self, fen, nodes):
        """Fixed-node search of @fen; returns the score in centipawns from @fen's side to move."""
        self.send(f"position fen {fen}")
        self.send(f"go nodes {nodes}")
        score = 0
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("engine died")
            mate = re.search(r"score mate (-?\d+)", line)
            centipawns = re.search(r"score cp (-?\d+)", line)
            if mate:
                moves_to_mate = int(mate.group(1))
                score = MATE_CP - abs(moves_to_mate) if moves_to_mate > 0 else -(MATE_CP - abs(moves_to_mate))
            elif centipawns:
                score = int(centipawns.group(1))
            if line.startswith("bestmove"):
                return score

    def quit(self):
        try:
            self.send("quit")
            self.process.wait(timeout=5)
        except Exception:
            self.process.kill()


class EnginePool:
    """A pool of persistent engines handed out to worker threads."""

    def __init__(self, binary, net, count):
        self.engines = queue.Queue()
        for _ in range(count):
            self.engines.put(Engine(binary, net))
        self.count = count

    def map_scores(self, jobs):
        """jobs: list of (fen, nodes). Returns scores in order, engine pool fanned out over threads."""
        results = [None] * len(jobs)

        def work(index, fen, nodes):
            engine = self.engines.get()
            try:
                results[index] = engine.search_nodes(fen, nodes)
            finally:
                self.engines.put(engine)

        threads = [threading.Thread(target=work, args=(i, fen, nodes)) for i, (fen, nodes) in enumerate(jobs)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return results

    def quit(self):
        while not self.engines.empty():
            self.engines.get_nowait().quit()


def run_cli(binary, *args):
    """One-shot zenith subcommand (moves / polykey / applymoves); returns stripped stdout."""
    result = subprocess.run([binary, *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{args[0]} failed: {result.stderr.strip()}")
    return result.stdout.strip()


# ---- the store -----------------------------------------------------------------------------------------
# store["positions"][polykey_hex] = {
#   "fen": str, "ply": int, "expanded": bool, "value": int (cp, side to move),
#   "moves": {uci: {"score": int, "child": polykey_hex|None, "weight": int, "playout": [points, games]}}
# }


def load_store(path):
    if os.path.exists(path):
        with open(path) as handle:
            return json.load(handle)
    return {"positions": {}, "waves": 0}


def save_store(store, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(store, handle)
    os.replace(temporary, path)


# ---- expansion -----------------------------------------------------------------------------------------


def expand_position(args, pool, store, fen, ply):
    """Screen/rescore/record one position. Returns its store entry (creating it)."""
    key = run_cli(args.engine, "polykey", fen)
    positions = store["positions"]
    if key in positions and positions[key]["expanded"]:
        return key
    legal_moves = run_cli(args.engine, "moves", fen).split()
    if not legal_moves:
        positions[key] = {"fen": fen, "ply": ply, "expanded": True, "value": 0, "moves": {}}
        return key

    child_fens = [run_cli(args.engine, "applymoves", fen, move) for move in legal_moves]
    screen_scores = pool.map_scores([(child, args.screen_nodes) for child in child_fens])
    scored = sorted(zip(legal_moves, child_fens, screen_scores), key=lambda item: item[2])  # child POV: low = good for parent

    # Keep the best few by screen (parent score = -child score), then rescore them deeper.
    kept = [item for item in scored if -item[2] >= -scored[0][2] - args.screen_margin][: args.max_candidates]
    rescored = pool.map_scores([(child, args.rescore_nodes) for _, child, _ in kept])
    candidates = sorted(
        ((move, child, -score) for (move, child, _), score in zip(kept, rescored)), key=lambda c: -c[2])
    best_score = candidates[0][2]
    candidates = [c for c in candidates if c[2] >= best_score - args.keep_margin]

    moves = {}
    for move, child, score in candidates:
        moves[move] = {"score": score, "child": None, "weight": 0, "playout": [0.0, 0]}
    positions[key] = {"fen": fen, "ply": ply, "expanded": True, "value": best_score, "moves": moves}
    return key


def backup_values(store):
    """Negamax the DAG: a parent's move score becomes -child value where the child is expanded."""
    positions = store["positions"]
    order = sorted(positions.values(), key=lambda p: -p["ply"])  # deepest first
    for position in order:
        best = None
        for move_data in position["moves"].values():
            child = move_data["child"]
            if child and child in positions and positions[child]["expanded"] and positions[child]["moves"]:
                move_data["score"] = -positions[child]["value"]
            if best is None or move_data["score"] > best:
                best = move_data["score"]
        if best is not None:
            position["value"] = best


def assign_weights(store, keep_margin):
    """Book weights per position: best move anchors at 100, others decay 2/cp; dropped beyond margin."""
    for position in store["positions"].values():
        if not position["moves"]:
            continue
        best = max(data["score"] for data in position["moves"].values())
        for data in position["moves"].values():
            gap = best - data["score"]
            playout_points, playout_games = data["playout"]
            playout_shift = 0.0
            if playout_games > 0:
                playout_shift = 30.0 * (playout_points / playout_games - 0.5)  # +/-15 weight swing
            data["weight"] = max(0, round(100 - 2 * gap + playout_shift)) if gap <= keep_margin else 0


def frontier(store, args):
    """Unexpanded (position, move) pairs reachable through book moves, by descending reach probability."""
    positions = store["positions"]
    reach = {}
    root_key = None
    for key, position in positions.items():
        if position["ply"] == 0:
            root_key = key
    if root_key is None:
        return []
    reach[root_key] = 1.0
    for key, position in sorted(positions.items(), key=lambda item: item[1]["ply"]):
        my_reach = reach.get(key, 0.0)
        if my_reach <= 0:
            continue
        total_weight = sum(d["weight"] for d in position["moves"].values()) or 1
        for move, data in position["moves"].items():
            if data["child"] and data["weight"] > 0:
                share = my_reach * data["weight"] / total_weight
                reach[data["child"]] = max(reach.get(data["child"], 0.0), share)
    pending = []
    for key, position in positions.items():
        if position["ply"] >= args.max_ply or abs(position["value"]) > args.abandon_cp:
            continue
        for move, data in position["moves"].items():
            if data["child"] is None and data["weight"] > 0:
                pending.append((reach.get(key, 0.0) * data["weight"] / 100.0, key, move))
    pending.sort(reverse=True)
    return pending


def playout_near_ties(args, store, key):
    """Self-play game pairs for candidates within --playout-margin of the best move of @key."""
    position = store["positions"][key]
    moves = position["moves"]
    if len(moves) < 2 or args.playout_games <= 0:
        return
    best = max(data["score"] for data in moves.values())
    tied = [(move, data) for move, data in moves.items() if best - data["score"] <= args.playout_margin]
    if len(tied) < 2:
        return
    for move, data in tied:
        if data["playout"][1] > 0:
            continue
        child_fen = run_cli(args.engine, "applymoves", position["fen"], move)
        white_points, games = playout_match(args, child_fen)
        # playout_match returns WHITE's points; the candidate mover is the side NOT to move in the child.
        mover_is_white = child_fen.split()[1] == "b"
        data["playout"] = [white_points if mover_is_white else games - white_points, games]


def playout_match(args, fen):
    """Zenith-vs-Zenith game pairs from @fen at jittered node budgets; returns (points, games) for
    the side to move of @fen. Node jitter makes each pair distinct despite a deterministic engine."""
    epd_path = os.path.join(os.path.dirname(args.state) or ".", "playout.epd")
    with open(epd_path, "w") as handle:
        handle.write(fen + "\n")
    total_points, total_games = 0.0, 0
    budgets = [args.playout_nodes + round(args.playout_nodes * 0.25 * i) for i in range(args.playout_games)]
    for budget in budgets:
        command = [args.fastchess,
                   "-engine", f"cmd={args.engine}", "name=first", "option.Threads=1", f"option.EvalFile={args.net}",
                   "-engine", f"cmd={args.engine}", "name=second", "option.Threads=1", f"option.EvalFile={args.net}",
                   "-each", "proto=uci", "tc=inf", f"nodes={budget}",
                   "-rounds", "1", "-games", "2", "-repeat",
                   "-openings", f"file={epd_path}", "format=epd",
                   "-draw", "movenumber=34", "movecount=6", "score=15",
                   "-resign", "movecount=4", "score=800",
                   "-concurrency", "2"]
        result = subprocess.run(command, capture_output=True, text=True)
        # Per-engine tallies are useless here (identical engines color-swap to ~50%); what measures the
        # POSITION is the per-game result by color. Count WHITE's points from the game lines.
        for outcome in re.findall(r"Finished game \d+[^:]*:\s*(1-0|0-1|1/2-1/2)", result.stdout):
            total_points += {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[outcome]
            total_games += 1
    return total_points, total_games


# ---- emission ------------------------------------------------------------------------------------------

CASTLE_TARGETS = {"e1g1": "e1h1", "e1c1": "e1a1", "e8g8": "e8h8", "e8c8": "e8a8"}


def encode_move(uci, is_castling):
    """Polyglot move field: to_file 0-2 | to_rank 3-5 | from_file 6-8 | from_rank 9-11 | promo 12-14."""
    if is_castling and uci in CASTLE_TARGETS:
        uci = CASTLE_TARGETS[uci]
    from_file, from_rank = ord(uci[0]) - 97, ord(uci[1]) - 49
    to_file, to_rank = ord(uci[2]) - 97, ord(uci[3]) - 49
    promotion = {"n": 1, "b": 2, "r": 3, "q": 4}.get(uci[5:6], 0)
    return to_file | to_rank << 3 | from_file << 6 | from_rank << 9 | promotion << 12


def is_castling_move(fen, uci):
    """King move e1g1/e1c1/e8g8/e8c8 counts as castling only if a king actually stands on the from-square."""
    if uci not in CASTLE_TARGETS:
        return False
    board = fen.split()[0]
    ranks = board.split("/")
    file_index, rank_index = ord(uci[0]) - 97, 8 - int(uci[1])
    row, column = ranks[rank_index], 0
    for ch in row:
        if ch.isdigit():
            column += int(ch)
        else:
            if column == file_index:
                return ch in "kK"
            column += 1
    return False


def emit_book(store, path):
    """Write the key-sorted big-endian Polyglot .bin from every positive-weight book move."""
    entries = []
    for key_hex, position in store["positions"].items():
        for move, data in position["moves"].items():
            if data["weight"] > 0 and position["moves"]:
                encoded = encode_move(move, is_castling_move(position["fen"], move))
                entries.append((int(key_hex, 16), encoded, data["weight"]))
    entries.sort()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as handle:
        for key, move, weight in entries:
            handle.write(struct.pack(">QHHI", key, move, weight, 0))
    log(f"emitted {len(entries)} entries ({os.path.getsize(path):,} bytes) -> {path}")


# ---- match gate ----------------------------------------------------------------------------------------


def book_match(args, book_path):
    """Book vs bookless from startpos; the book side's weighted-random picks provide game variety."""
    command = [args.fastchess,
               "-engine", f"cmd={args.engine}", "name=book", "option.Threads=1", f"option.EvalFile={args.net}",
               "option.OwnBook=true", f"option.BookFile={book_path}",
               "-engine", f"cmd={args.engine}", "name=nobook", "option.Threads=1", f"option.EvalFile={args.net}",
               "-each", "proto=uci", f"tc={args.match_tc}",
               "-rounds", str(max(1, args.games // 2)), "-games", "2", "-repeat",
               "-draw", "movenumber=40", "movecount=8", "score=20",
               "-resign", "movecount=3", "score=600",
               "-concurrency", str(args.concurrency)]
    log(f"match: book vs bookless, {args.games} games at {args.match_tc}")
    result = subprocess.run(command, capture_output=True, text=True)
    elo = re.findall(r"Elo difference:\s*(-?[\d.]+)\s*\+/-\s*([\d.]+)", result.stdout)
    tallies = re.findall(r"Wins:\s*(\d+),\s*Losses:\s*(\d+),\s*Draws:\s*(\d+)", result.stdout)
    if tallies:
        wins, losses, draws = tallies[-1]
        log(f"match result (book side): W{wins} L{losses} D{draws}" +
            (f"  Elo {elo[-1][0]} +/- {elo[-1][1]}" if elo else ""))
    else:
        log("match produced no tally — output follows"); print(result.stdout[-2000:])


# ---- driver --------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="./build/zenith")
    parser.add_argument("--net", default="nets/zenith-ob2.nnue")
    parser.add_argument("--fastchess", default=os.path.expanduser("~/pawnstar_nnue/fastchess/fastchess"))
    parser.add_argument("--state", default="data/book/store.json")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--expand", type=int, default=0, help="expand up to N new positions this run")
    parser.add_argument("--max-ply", type=int, default=8)
    parser.add_argument("--screen-nodes", type=int, default=150_000)
    parser.add_argument("--rescore-nodes", type=int, default=2_000_000)
    parser.add_argument("--screen-margin", type=int, default=60, help="cp window into rescoring")
    parser.add_argument("--keep-margin", type=int, default=40, help="cp window into the book")
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument("--abandon-cp", type=int, default=150, help="stop expanding lines this far gone")
    parser.add_argument("--playout-margin", type=int, default=15)
    parser.add_argument("--playout-games", type=int, default=0, help="game pairs per near-tie candidate (0 = off)")
    parser.add_argument("--playout-nodes", type=int, default=50_000)
    parser.add_argument("--emit", default=None, help="write the Polyglot .bin here after expanding")
    parser.add_argument("--match", default=None, help="play the book-vs-bookless gate with this .bin")
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--match-tc", default="8+0.08")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    store = load_store(args.state)
    if args.expand > 0:
        pool = EnginePool(args.engine, args.net, args.workers)
        try:
            if not store["positions"]:
                expand_position(args, pool, store, START_FEN, 0)
                backup_values(store)
                assign_weights(store, args.keep_margin)
                save_store(store, args.state)
            expanded = 0
            while expanded < args.expand:
                pending = frontier(store, args)
                if not pending:
                    log("frontier empty — raise --max-ply or margins")
                    break
                batch = pending[: min(args.workers * 2, args.expand - expanded)]
                for _, parent_key, move in batch:
                    parent = store["positions"][parent_key]
                    child_fen = run_cli(args.engine, "applymoves", parent["fen"], move)
                    child_key = expand_position(args, pool, store, child_fen, parent["ply"] + 1)
                    parent["moves"][move]["child"] = child_key
                    expanded += 1
                backup_values(store)
                assign_weights(store, args.keep_margin)
                if args.playout_games > 0:
                    for _, parent_key, _ in batch:
                        playout_near_ties(args, store, parent_key)
                    assign_weights(store, args.keep_margin)
                store["waves"] += 1
                save_store(store, args.state)
                total = len(store["positions"])
                log(f"wave {store['waves']}: +{len(batch)} expanded ({expanded}/{args.expand} this run, "
                    f"{total} positions total)")
        finally:
            pool.quit()

    if args.emit:
        backup_values(store)
        assign_weights(store, args.keep_margin)
        save_store(store, args.state)
        emit_book(store, args.emit)
    if args.match:
        book_match(args, args.match)


if __name__ == "__main__":
    main()
