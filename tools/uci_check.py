#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonny Reckless
"""The front-end gate: the engine's UCI protocol and its notation subcommands, driven through pipes.

The engine's own gates (perft, bench, legalcheck, ...) test the chess inside one process; this one tests what a
GUI, a book builder or a test harness actually talks to, by running the binary as they do:

- the notation subcommands the book builder uses (moves, polykey, san, applymoves), against hand-verified answers -
  pinned-pawn promotion captures, the Polyglot specification's keys, every SAN disambiguation form;
- the NNUE paths outside nnuecheck: nnueeval (and the net's colour symmetry), the loader's verdicts on a bad magic,
  a truncated and an over-long file, and the two legacy formats it still reads;
- every UCI option: Hash, Clear Hash, Threads, Move Overhead, EvalFile (a file, a missing file, the embedded net),
  OwnBook and BookFile (a file, a missing, empty, ragged or unsorted one, a path with a space, castling entries),
  and the tunable search parameters, whose effect is shown by the node count of a fixed-depth search;
- the position and go forms: fen and startpos with moves, an illegal move, a history past its cap, a half-move
  clock past 255 (still printed whole, still a fifty-move draw) and FEN clocks out of range, the clock counting
  through castling (neither a pawn move nor a capture) and reset by a capture and en passant, the fifty-move draw
  reached by castling, a clock, a fallen flag, a node budget, movetime, perft, infinite with stop, unknown tokens, a
  mate, a mated position;
- malformed input, after which the engine must still answer isready.

With --terminal it also runs legalcheck on a pseudo-terminal (POSIX only), resizing it as the gate runs, and holds
the in-place progress line to its contract: drawn with its time column on a wide terminal, without it on a narrow
one, and always cleared before a result line (the gate ends on a terminal too narrow for any frame, the fallback that
draws none).

Usage: python3 tools/uci_check.py [--terminal] <engine>   (run from anywhere; exit status 1 on any failure)
"""
import os
import queue
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NET = "nets/zenith-cap2.nnue"
BOOK = "books/zenith-book-r4.bin"
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
TIMEOUT = 120  # seconds any one engine process may take: generous for an instrumented build on a slow runner


class Gate:
    """Runs the engine and prints one [PASS]/[FAIL] line per check, on the engine gates' column grid."""

    def __init__(self, engine):
        self.engine = engine
        self.passed = 0
        self.failed = 0

    def report(self, is_pass, detail, aux, tail):
        """Print one result line: name, detail, the node column left blank, aux, the time and Mnps blank, tail."""
        tag = "[PASS]" if is_pass else "[FAIL]"
        print(f"{tag} {'uci':<6} {detail:<9} {'':>21} {aux:<25} {'':>9} {'':>11} {tail}", flush=True)
        if is_pass:
            self.passed += 1
        else:
            self.failed += 1
        return is_pass

    def run(self, *args, stdin=""):
        """Run the engine with @p args to completion; return (exit status, stdout, stderr)."""
        result = subprocess.run([self.engine, *args], input=stdin, capture_output=True, text=True, encoding="utf-8",
                                errors="replace", cwd=ROOT, timeout=TIMEOUT)
        return result.returncode, result.stdout, result.stderr

    def session(self, steps):
        """Drive one UCI session: each step is a line to send, or (line, regex) to send and then wait for an output
        line matching regex, or a number of seconds to sleep. A final quit is sent. Return every output line."""
        process = subprocess.Popen([self.engine], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace", cwd=ROOT)
        lines = []
        arrived = queue.Queue()

        def reader():
            for line in process.stdout:
                arrived.put(line.rstrip("\r\n"))
            arrived.put(None)

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        deadline = time.monotonic() + TIMEOUT

        def wait_for(pattern):
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"no line matching {pattern!r}")
                line = arrived.get(timeout=remaining)
                if line is None:
                    raise EOFError(f"engine exited before a line matching {pattern!r}")
                lines.append(line)
                if re.search(pattern, line):
                    return

        try:
            for step in steps:
                if isinstance(step, (int, float)):
                    time.sleep(step)
                    continue
                text, pattern = (step, None) if isinstance(step, str) else step
                process.stdin.write(text + "\n")
                process.stdin.flush()
                if pattern is not None:
                    wait_for(pattern)
            process.stdin.write("quit\n")
            process.stdin.close()
            process.wait(timeout=max(1.0, deadline - time.monotonic()))
        except (TimeoutError, EOFError, queue.Empty, subprocess.TimeoutExpired, BrokenPipeError) as error:
            process.kill()
            process.wait()
            lines.append(f"<session failed: {error}>")
        thread.join(timeout=5)
        while not arrived.empty():
            line = arrived.get()
            if line is not None:
                lines.append(line)
        lines.append(f"<exit {process.returncode}>")
        return lines


def go(command):
    """A go step that waits for its bestmove."""
    return (command, r"^bestmove ")


def last_info_nodes(lines):
    """The node count of the last `info depth` line, or None."""
    nodes = None
    for line in lines:
        match = re.match(r"info depth \d+ .*\bnodes (\d+)", line)
        if match:
            nodes = int(match.group(1))
    return nodes


def bestmoves(lines):
    """Every bestmove a session printed, in order."""
    return [line.split()[1] for line in lines if line.startswith("bestmove ")]


def check_notation(gate):
    """The book builder's subcommands: moves, polykey, san, applymoves, against hand-verified answers."""
    status, out, _ = gate.run("moves", START_FEN)
    expected = {"a2a3", "a2a4", "b2b3", "b2b4", "c2c3", "c2c4", "d2d3", "d2d4", "e2e3", "e2e4", "f2f3", "f2f4",
                "g2g3", "g2g4", "h2h3", "h2h4", "b1a3", "b1c3", "g1f3", "g1h3"}
    gate.report(status == 0 and set(out.split()) == expected and len(out.split()) == 20, "moves",
                f"{len(out.split())} legal moves", START_FEN)

    # A pawn pinned on the long diagonal may only take its pinner, and promotes as it does; the push is off the ray.
    fen = "b3k3/1P6/8/8/8/8/8/7K w - - 0 1"
    status, out, _ = gate.run("moves", fen)
    expected = {"b7a8q", "b7a8r", "b7a8b", "b7a8n", "h1g1", "h1g2", "h1h2"}
    gate.report(status == 0 and set(out.split()) == expected, "moves", "pinned capture-promotion", fen)

    fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"  # fool's mate: no legal move
    status, out, _ = gate.run("moves", fen)
    gate.report(status == 0 and out.strip() == "", "moves", "none (checkmated)", fen)

    status_bare, _, err_bare = gate.run("moves")
    status_bad, _, err_bad = gate.run("moves", "not a fen")
    gate.report(status_bare == 1 and "usage" in err_bare and status_bad == 1 and "invalid FEN" in err_bad, "moves",
                "usage, bad FEN refused", "(moves with no FEN, and with a malformed one)")

    # The Polyglot specification's own test vectors.
    status, out, _ = gate.run("polykey", START_FEN)
    status2, out2, _ = gate.run("polykey", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    gate.report(status == 0 and out.strip() == "463b96181691fc9c" and status2 == 0 and
                out2.strip() == "823c9b50fd114196", "polykey", "spec vectors", "(startpos and 1. e4)")
    status_bare, _, _ = gate.run("polykey")
    status_bad, _, _ = gate.run("polykey", "8/8/8 w")
    gate.report(status_bare == 1 and status_bad == 1, "polykey", "usage, bad FEN refused",
                "(polykey with no FEN, and with a malformed one)")

    san_cases = [
        (START_FEN, ["e2e4", "d7d5", "e4d5", "d8d5"], "e4 d5 exd5 Qxd5"),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", ["e1g1", "e8c8"], "O-O O-O-O"),
        ("4k3/8/8/8/8/5N2/8/1N2K3 w - - 0 1", ["b1d2"], "Nbd2"),
        ("4k3/8/8/R7/8/8/8/R3K3 w - - 0 1", ["a1a3"], "R1a3"),
        ("4k3/8/8/8/8/Q7/8/Q1Q1K3 w - - 0 1", ["a1b2"], "Qa1b2"),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", ["e5d6"], "exd6"),
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", ["a7a8q"], "a8=Q+"),
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", ["a7a8n"], "a8=N"),
        ("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", ["a1a8"], "Ra8#"),
    ]
    for fen, moves, want in san_cases:
        status, out, _ = gate.run("san", fen, *moves)
        gate.report(status == 0 and out.strip() == want, "san", out.strip()[:25], fen)
    status_bare, _, _ = gate.run("san", START_FEN)
    status_bad, _, _ = gate.run("san", "nonsense", "e2e4")
    status_illegal, _, err = gate.run("san", START_FEN, "e2e5")
    gate.report(status_bare == 1 and status_bad == 1 and status_illegal == 1 and "illegal move" in err, "san",
                "usage, bad input refused", "(no move; a malformed FEN; an illegal move)")

    status, out, _ = gate.run("applymoves", START_FEN, "e2e4")
    status2, out2, _ = gate.run("applymoves", START_FEN, "e2e4", "e7e5", "e1e2")
    status3, out3, _ = gate.run("applymoves", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    gate.report(status == 0 and out.strip() == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1" and
                status2 == 0 and out2.strip() == "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPPKPPP/RNBQ1BNR b kq - 1 2" and
                status3 == 0 and out3.strip() == "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "applymove",
                "FENs as expected", "(1. e4; 1. e4 e5 2. Ke2; no moves)")
    status_bare, _, _ = gate.run("applymoves")
    status_bad, _, _ = gate.run("applymoves", "x")
    status_illegal, _, err = gate.run("applymoves", START_FEN, "e2e4", "e2e4")
    gate.report(status_bare == 1 and status_bad == 1 and status_illegal == 1 and "illegal move" in err, "applymove",
                "usage, bad input refused", "(no FEN; a malformed FEN; an illegal move)")


def write_bytes(path, data):
    with open(os.path.join(ROOT, path), "wb") as out:
        out.write(data)


def check_nnue(gate, scratch):
    """nnueeval, the net's colour symmetry, the loader's verdicts, and the two legacy formats."""
    with open(os.path.join(ROOT, NET), "rb") as source:
        net = source.read()
    if net[:8] != b"ZNNUE5\x00\x00":
        gate.report(False, "nnue", "unexpected net format", NET)
        return
    # The body is the feature transformer (weights then bias, int16), then per bucket 2 x hidden int16 output
    # weights, then one int32 bias per bucket: 8 x 768 x H + H + 8 x 2H int16 + 8 int32 = 2 x (6152 H + 16) + 32.
    buckets = 8
    hidden = (len(net) - 8 - 4 * buckets) // (2 * (8 * 768 + 1 + 2 * buckets))
    transformer_end = 8 + 2 * (8 * 768 * hidden + hidden)
    first_head = net[transformer_end:transformer_end + 4 * hidden]
    first_bias = net[transformer_end + 2 * 2 * hidden * buckets:transformer_end + 2 * 2 * hidden * buckets + 4]
    legacy_v3 = b"ZNNUE3\x00\x00" + net[8:transformer_end] + first_head + first_bias
    legacy_v4 = b"ZNNUE4\x00\x00" + net[8:]
    files = {
        "v3": (legacy_v3, None),
        "v4": (legacy_v4, None),
        "bad magic": (b"NOTANNUE" + net[8:], "bad magic"),
        "truncated": (net[:-1], "truncated data"),
        "over-long": (net + b"\x00", "size mismatch"),
        "empty": (b"", None),
    }
    paths = {}
    for name, (data, _) in files.items():
        paths[name] = f"{scratch}/{name.replace(' ', '-')}.nnue"
        write_bytes(paths[name], data)

    # A position and its colour mirror (ranks flipped, colours swapped, the other side to move) must evaluate alike.
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    mirror = "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR b KQkq - 2 3"
    status, out, _ = gate.run("nnueeval", NET, stdin=f"{START_FEN}\n\nnot a fen\n{fen}\n{mirror}\n")
    values = out.split()
    is_ok = (status == 0 and len(values) == 4 and all(re.fullmatch(r"-?\d+", value) for value in values) and
             values[1] == "0" and values[2] == values[3])
    gate.report(is_ok, "nnueeval", f"{' '.join(values)}"[:25], "(startpos, blank skipped, bad FEN -> 0, mirror pair)")

    for name in ("v3", "v4"):
        status, out, _ = gate.run("nnueeval", paths[name], stdin=f"{fen}\n{mirror}\n")
        values = out.split()
        gate.report(status == 0 and len(values) == 2 and values[0] == values[1], "nnueeval",
                    f"legacy {name} loads", f"(ZNNUE{name[1]} read from {NET}, mirror pair {' '.join(values)})")

    for name in ("bad magic", "truncated", "over-long", "empty"):
        status, _, err = gate.run("nnueeval", paths[name], stdin=f"{START_FEN}\n")
        message = files[name][1]
        gate.report(status == 1 and "failed to load" in err and (message is None or message in err), "loader",
                    f"{name} refused", f"({message or 'no bytes'})")
    status, _, err = gate.run("nnueeval", f"{scratch}/missing.nnue")
    gate.report(status == 1 and "failed to load" in err, "loader", "missing file refused", "(nnueeval)")
    status, _, err = gate.run("nnuecheck", f"{scratch}/missing.nnue")
    gate.report(status == 1 and "failed to load" in err, "nnuecheck", "missing file refused", "(nnuecheck)")
    status_eval, _, _ = gate.run("nnueeval")
    status_check, _, _ = gate.run("nnuecheck")
    gate.report(status_eval == 1 and status_check == 1, "usage", "no net named", "(nnueeval, nnuecheck)")
    return paths


def check_cli_misc(gate):
    status, out, _ = gate.run("bench", "2")
    gate.report(status == 0 and "informational" in out and out.count("bench") >= 9, "bench",
                "depth 2 informational", "(no pinned counts at this depth, so no verdict)")
    status, out, err = gate.run("frobnicate", stdin="isready\nquit\n")
    gate.report(status == 0 and "unknown subcommand" in err and "readyok" in out, "argv", "unknown -> UCI loop",
                "(a GUI's stray argument starts the protocol)")


def check_options(gate, nnue_paths):
    """Every UCI option, and that each takes effect."""
    lines = gate.session([("uci", r"^uciok$"), ("isready", r"^readyok$")])
    names = {match.group(1) for line in lines for match in [re.match(r"option name (.+?) type ", line)] if match}
    wanted = {"Hash", "Threads", "Move Overhead", "Clear Hash", "EvalFile", "OwnBook", "BookFile", "RfpMargin",
              "NmpDivisor", "LmpBase", "FutilityBase", "FutilityMargin", "SeeCaptureMargin", "LmrBase", "LmrDivisor",
              "SingularMargin", "AspirationDelta", "HistoryMax"}
    gate.report(names == wanted and "readyok" in lines and any(line.startswith("id name Zenith") for line in lines),
                "uci", f"{len(names)} options", "(id, every option, uciok, readyok)")

    # A fixed-depth single-threaded search from a fresh process is deterministic, so its node count shows whether
    # an option reached the search: a tuned parameter moves it, an unknown name or a non-numeric value does not.
    def nodes_with(*options):
        steps = ["setoption name Threads value 1", *options, "position startpos moves e2e4 e7e5", go("go depth 9")]
        return last_info_nodes(gate.session(steps))

    baseline = nodes_with()
    tuned = nodes_with("setoption name RfpMargin value 200")
    lmr = nodes_with("setoption name LmrDivisor value 400")
    ignored = nodes_with("setoption name NoSuchOption value 5", "setoption name RfpMargin value abc",
                         "setoption name Ponder value true", "setoption name Move Overhead value 30",
                         "setoption name Clear Hash")
    gate.report(baseline is not None and tuned is not None and tuned != baseline, "setoption",
                f"RfpMargin {baseline}->{tuned}", "(a tuned search parameter changes the search)")
    gate.report(lmr is not None and lmr != baseline, "setoption", f"LmrDivisor {baseline}->{lmr}",
                "(rebuilds the reduction table)")
    gate.report(ignored == baseline, "setoption", f"ignored: {ignored}",
                "(unknown name, non-numeric value, Clear Hash, Move Overhead leave a fixed-depth search alone)")
    lines = gate.session(["setoption name Hash value 0", "position startpos", go("go depth 6"),
                          "setoption name Hash value 16", go("go depth 6")])
    gate.report(len(bestmoves(lines)) == 2, "Hash", "0 -> 1 MB, then 16 MB", "(a resized table searches)")

    # The embedded net and the same net loaded from its file search identically; a missing file falls back to it.
    from_file = nodes_with(f"setoption name EvalFile value {NET}")
    gate.report(from_file == baseline, "EvalFile", f"file = embedded: {from_file}", f"({NET})")
    lines = gate.session(["setoption name Threads value 1",
                          (f"setoption name EvalFile value {NET}", r"^info string loaded NNUE"),
                          ("setoption name EvalFile value nets/missing.nnue", r"^info string failed to load NNUE"),
                          ("setoption name EvalFile value", r"^info string using the embedded net"),
                          (f"setoption name EvalFile value {nnue_paths['v4']}", r"^info string loaded NNUE"),
                          "position startpos", go("go depth 4")])
    gate.report(len(bestmoves(lines)) == 1, "EvalFile", "load, fail, reset", "(a file, a missing file, empty -> "
                "the embedded net, a legacy net that then searches)")

    # 999 is clamped to 256 but not searched with: 256 searchers would take some 600 MB on a shared runner.
    lines = gate.session(["setoption name Threads value 0", "position startpos", go("go depth 6"),
                          "setoption name Threads value 999", "setoption name Threads value 3", go("go depth 6")])
    gate.report(len(bestmoves(lines)) == 2, "Threads", "clamped, pool resized", "(0 -> one thread, then three)")


def polyglot_entry(key, move_from, move_to, weight=1):
    """One 16-byte big-endian Polyglot record; castling is stored king-takes-rook."""
    def square(name):
        return (ord(name[0]) - ord("a")) | ((int(name[1]) - 1) << 3)

    move = square(move_to) | (square(move_from) << 6)
    return struct.pack(">QHHI", key, move, weight, 0)


def check_books(gate, scratch):
    """OwnBook and BookFile: the embedded book, crafted books, and the books the loader must refuse."""
    def key(fen):
        status, out, _ = gate.run("polykey", fen)
        return int(out.strip(), 16) if status == 0 else 0

    white = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
    black = "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"
    entries = sorted([(key(white), polyglot_entry(key(white), "e1", "h1")),
                      (key(black), polyglot_entry(key(black), "e8", "a8"))])
    castles = b"".join(entry for _, entry in entries)
    good = f"{scratch}/castles.bin"
    spaced = f"{scratch}/castles book.bin"
    ragged = f"{scratch}/ragged.bin"
    empty = f"{scratch}/empty.bin"
    unsorted = f"{scratch}/unsorted.bin"
    write_bytes(good, castles)
    write_bytes(spaced, castles)
    write_bytes(ragged, castles[:-1])
    write_bytes(empty, b"")
    write_bytes(unsorted, castles[16:] + castles[:16])

    lines = gate.session([(f"setoption name BookFile value {good}", r"^info string (loaded|failed)"),
                          "setoption name OwnBook value true",
                          f"position fen {white}", go("go wtime 10000 btime 10000"),
                          f"position fen {black}", go("go wtime 10000 btime 10000 winc 100 binc 100")])
    gate.report(bestmoves(lines) == ["e1g1", "e8c8"] and sum("book move" in line for line in lines) == 2, "BookFile",
                "castles from the book", "(Polyglot king-takes-rook entries -> O-O and O-O-O)")

    lines = gate.session([(f"setoption name BookFile value {spaced}", r"^info string (loaded|failed)")])
    gate.report(any(line == f"info string loaded book {spaced}" for line in lines), "BookFile", "a path with a space",
                "(the value is every token after `value`)")

    refused = []
    for path in (f"{scratch}/missing.bin", ragged, empty, unsorted):
        lines = gate.session([(f"setoption name BookFile value {path}", r"^info string (loaded|failed)")])
        refused.append(any(line.startswith("info string failed to load book") for line in lines))
    gate.report(all(refused), "BookFile", f"{sum(refused)}/4 refused", "(missing, not whole records, empty, "
                "unsorted)")

    lines = gate.session([("setoption name OwnBook value true", r"^info string using the embedded book"),
                          "position startpos", go("go wtime 60000 btime 60000 movestogo 40"),
                          "position fen 8/8/4k3/8/8/4K3/8/8 w - - 0 1", go("go wtime 1000 btime 1000"),
                          "setoption name OwnBook value false", "position startpos", go("go depth 2")])
    books = [line for line in lines if "book move" in line]
    gate.report(len(bestmoves(lines)) == 3 and len(books) == 1, "OwnBook", "embedded, miss, off",
                "(book move at startpos; a search when out of book or off)")


def check_positions_and_go(gate):
    """The position and go forms, and what each must print."""
    lines = gate.session(["position fen 8/8/8/8/8/8/8/K6k w - - 5x 3", ("d", r"/"),
                          f"position fen {START_FEN} moves e2e4 e7e5 g1f3", ("d", r"/"),
                          "position startpos moves e2e4 e9e5 d2d4", ("d", r"/"),
                          "position fen rnbqkbnr/pppppppp/8/8 w KQkq - 0 1", ("d", r"/"),
                          "position", "position banana", ("d", r"/")])
    fens = [line for line in lines if "/" in line and not line.startswith("<")]
    want = ["8/8/8/8/8/8/8/K6k w - - 5 0", "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"]
    gate.report(fens == want, "position", "fen, moves, refusals",
                "(trailing junk in a clock; an illegal move stops the replay; a bad FEN is ignored)")
    if fens != want:
        print("       got:", fens, flush=True)

    # 8100 knight moves: the replay stops at the history cap (8192 less MAX_PLY headroom = 8064 moves, a whole
    # number of four-move cycles, so the knights are home on move 4033 with all 8064 plies on the half-move clock -
    # a byte-wide clock printed 128), and the search still answers.
    shuffle = " ".join(["g1f3", "g8f6", "f3g1", "f6g8"] * 2025)
    lines = gate.session([f"position startpos moves {shuffle}", ("d", r"/"), go("go depth 3")])
    board = START_FEN.rsplit(" ", 2)[0]
    gate.report(f"{board} 8064 4033" in lines and len(bestmoves(lines)) == 1, "position", "history capped at 8064",
                "(8100 moves: the replay keeps MAX_PLY headroom; the clock reads 8064)")

    # The fifty-move draw past a byte's worth of reversible plies: 300 rook and king moves bring K+R v K home with
    # White to move and the clock at 300, so every reply is a fifty-move draw and the search scores 0 at every
    # depth. A clock that wrapped at 256 read 44 here, and the search saw a won ending instead.
    shuffle = " ".join(["a1a2", "e8d8", "a2a1", "d8e8"] * 75)
    lines = gate.session(["setoption name Threads value 1", f"position fen 4k3/8/8/8/8/8/8/R3K3 w - - 0 1 moves "
                          f"{shuffle}", ("d", r"/"), go("go depth 6")])
    scores = [match.group(1) for line in lines for match in [re.match(r"info depth \d+ .*\bscore (\w+ -?\d+)", line)]
              if match]
    gate.report("4k3/8/8/8/8/8/8/R3K3 w - - 300 151" in lines and len(scores) >= 6 and set(scores) == {"cp 0"},
                "position", f"300 plies: {scores[-1] if scores else '-'}", "(clock 300: the fifty-move draw, every "
                "depth)")

    # A FEN clock is any integer to the specification; the engine keeps one up to 10^9 and clamps the rest into
    # 0..10^9, so it neither wraps nor overflows as moves are made on it.
    lines = gate.session(["position fen 8/8/8/8/8/8/8/K6k w - - 300 3", ("d", r"/"),
                          "position fen 8/8/8/8/8/8/8/K6k w - - -7 3", ("d", r"/"),
                          "position fen 8/8/8/8/8/8/8/K6k w - - 99999999999 3 moves a1a2", ("d", r"/"),
                          go("go depth 2")])
    fens = [line for line in lines if "/" in line and not line.startswith("<")]
    want = ["8/8/8/8/8/8/8/K6k w - - 300 3", "8/8/8/8/8/8/8/K6k w - - 0 3", "8/8/8/8/8/8/K7/7k b - - 1000000001 3"]
    gate.report(fens == want and len(bestmoves(lines)) == 1, "position", "FEN clocks 300, -7, 1e11",
                "(kept, clamped to 0, clamped to 10^9 then a move)")
    if fens != want:
        print("       got:", fens, flush=True)

    # The fifty-move rule (FIDE 9.3) counts moves "without the movement of any pawn and without any capture", and
    # FEN's clock is the plies since the last of either. Castling is neither, so the clock counts through it,
    # irreversible as it is: the castling rights in the key keep the repetition scan from matching across it, not
    # the clock. Both sides castle, each way, on a clock of 7; then a capture and an en-passant capture reset it.
    lines = gate.session(["position fen r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 7 10 moves e1g1", ("d", r"/"),
                          "position fen r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 7 10 moves e1g1 e8c8", ("d", r"/"),
                          "position fen r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 7 10 moves e1c1 e8g8", ("d", r"/"),
                          "position fen r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 7 10 moves a1a8", ("d", r"/"),
                          "position fen 4k3/8/8/8/3p4/8/4P3/4K3 w - - 20 40 moves e2e4 d4e3", ("d", r"/")])
    fens = [line for line in lines if "/" in line and not line.startswith("<")]
    want = ["r3k2r/8/8/8/8/8/8/R4RK1 b kq - 8 10", "2kr3r/8/8/8/8/8/8/R4RK1 w - - 9 11",
            "r4rk1/8/8/8/8/8/8/2KR3R w - - 9 11", "R3k2r/8/8/8/8/8/8/4K2R b Kk - 0 10",
            "4k3/8/8/8/8/4p3/8/4K3 w - - 0 41"]
    gate.report(fens == want, "position", "clock through castling", "(O-O and O-O-O count 8, 9; a capture and en "
                "passant reset it)")
    if fens != want:
        print("       got:", fens, flush=True)

    # The fifty-move draw reached by castling, replayed and in the tree. K+R v K, a won ending (-1301 cp for Black
    # at depth 6 on a fresh clock). Replayed: O-O on a clock of 99 makes it 100, so Black is in a drawn position and
    # scores 0 at every depth. In the tree: White to move on 99, every move - O-O among them - makes the clock 100,
    # so White scores 0 at every depth; a castle that reset the clock would score O-O as a win.
    lines = gate.session(["setoption name Threads value 1", "position fen 4k3/8/8/8/8/8/8/4K2R w K - 99 80 moves e1g1",
                          ("d", r"/"), go("go depth 6"), "position fen 4k3/8/8/8/8/8/8/4K2R w K - 99 80",
                          go("go depth 6")])
    scores = [match.group(1) for line in lines for match in [re.match(r"info depth \d+ .*\bscore (\w+ -?\d+)", line)]
              if match]
    gate.report("4k3/8/8/8/8/8/8/5RK1 b - - 100 80" in lines and len(scores) >= 12 and set(scores) == {"cp 0"},
                "position", f"castling to 100: {scores[-1] if scores else '-'}", "(O-O on 99 makes 100: drawn, "
                "replayed and in the tree)")

    lines = gate.session(["setoption name Threads value 1", "position startpos moves e2e4",
                          go("go wtime 3000 btime 3000 winc 20 binc 20 movestogo 30"),
                          go("go wtime 0 btime 0"),
                          go("go nodes 20000"),
                          go("go movetime 150 depth"),
                          go("go ponder searchmoves e7e5 movetime 100"),
                          ("go perft 3", r"^nodes \d+"),
                          ("ucinewgame", None), ("isready", r"^readyok$"),
                          ("go perft 2", r"^nodes \d+")])
    perfts = [int(match.group(1)) for line in lines for match in [re.match(r"nodes (\d+)", line)] if match]
    divide = [line for line in lines if re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?: \d+", line)]
    gate.report(len(bestmoves(lines)) == 5 and perfts == [13160, 400] and len(divide) == 20 + 20, "go",
                f"5 searches, {len(perfts)} perfts", f"(clock, flag fall, nodes, movetime, unknown tokens; perft "
                f"divide {perfts[0] if perfts else '-'} after 1. e4, then 400)")

    lines = gate.session(["setoption name Threads value 2", "position startpos", ("go infinite", r"^info depth 4 "),
                          ("stop", r"^bestmove "), "stop", ("isready", r"^readyok$")])
    gate.report(len(bestmoves(lines)) == 1 and "readyok" in lines, "go", "infinite, then stop",
                "(stop ends an infinite search; a second stop is harmless)")

    lines = gate.session(["setoption name Threads value 1", "position fen 6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1",
                          go("go wtime 10000 btime 10000"),
                          "position fen k7/8/1K6/8/8/8/8/7R b - - 0 1", go("go depth 8"),
                          "position fen rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
                          go("go depth 3")])
    has_mate = any(re.search(r" score mate 1 .* pv a1a8", line) for line in lines)
    is_mated = any(re.search(r"^info depth 8 .* score mate -1 .* pv a8b8 h1h8", line) for line in lines)
    gate.report(has_mate and is_mated and bestmoves(lines) == ["a1a8", "a8b8", "0000"], "go", "mate 1, mated in 1",
                "(a mate score ends a clocked search; Kb8 Rh8#; no move -> bestmove 0000)")

    # A fifty-move clock older than the history the engine was given: the repetition scan starts at the root.
    lines = gate.session(["setoption name Threads value 1", "position fen 4k3/8/8/8/8/8/4P3/4K3 w - - 40 60",
                          go("go depth 6")])
    gate.report(len(bestmoves(lines)) == 1, "go", "clock past history", "(half-move clock 40 with no moves given)")


def check_malformed(gate):
    junk = ["", "   ", "\t", "foo bar", "setoption", "setoption name", "setoption name Hash", "position fen",
            "position fen 8/8/8 w", "position startpos moves", "d d d", "x" * 70000]
    lines = gate.session([*junk, ("isready", r"^readyok$"), ("position startpos", None), go("go depth 2")])
    gate.report("readyok" in lines and len(bestmoves(lines)) == 1 and lines[-1] == "<exit 0>", "malformed",
                "still answers", "(blank, unknown, truncated and oversized commands)")


def check_terminal(gate):
    """legalcheck on a pseudo-terminal: the in-place progress line drawn wide, narrow, or not at all, and cleared."""
    if os.name != "posix":
        print(f"[SKIP] {'uci':<6} {'terminal':<9} {'':>21} {'no pseudo-terminals here':<25}", flush=True)
        return
    import fcntl
    import pty
    import termios

    def set_width(fd, columns):
        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 24, columns, 0, 0))

    controller, terminal = pty.openpty()
    set_width(terminal, 200)
    process = subprocess.Popen([gate.engine, "legalcheck"], stdin=subprocess.DEVNULL, stdout=terminal,
                               stderr=subprocess.DEVNULL, cwd=ROOT)
    os.close(terminal)
    output = b""

    def frames():
        return re.findall(r"\r\[ \.\. \] ([^\r\x1b]*)\x1b\[K", output.decode("utf-8", "replace"))

    def is_wide(frame):
        return re.search(r"\d\.\d{3}s$", frame) is not None

    # 200 columns until a frame with its time column arrives, then 60 (room for the frame without it) until one of
    # those arrives, then 20, where no frame fits and none may be drawn.
    stage = 0
    while True:
        try:
            chunk = os.read(controller, 65536)
        except OSError:
            break
        if not chunk:
            break
        output += chunk
        if stage == 0 and any(is_wide(frame) for frame in frames()):
            set_width(controller, 60)
            stage = 1
        elif stage == 1 and any(frame.endswith("nodes") for frame in frames()):
            set_width(controller, 20)
            stage = 2
    os.close(controller)
    status = process.wait(timeout=TIMEOUT)
    text = output.decode("utf-8", "replace")
    wide = [frame for frame in frames() if is_wide(frame)]
    narrow = [frame for frame in frames() if frame.endswith("nodes")]
    results = list(re.finditer(r"\[PASS\] legal", text))
    # A result line may share its terminal line with a progress frame only if the frame was erased just before it.
    is_cleared = True
    for match in results:
        line_so_far = text[text.rfind("\n", 0, match.start()) + 1:match.start()]
        is_cleared &= "[ .. ]" not in line_so_far or line_so_far.endswith("\r\x1b[K")
    gate.report(status == 0 and stage == 2 and wide and narrow and len(results) == 9 and is_cleared, "terminal",
                f"{len(wide)} wide, {len(narrow)} narrow",
                "(legalcheck on a pty: progress with and without time, cleared before each result)")


def main():
    arguments = sys.argv[1:]
    is_terminal = "--terminal" in arguments
    arguments = [argument for argument in arguments if argument != "--terminal"]
    if len(arguments) != 1:
        raise SystemExit("usage: uci_check.py [--terminal] <engine>")
    engine = os.path.abspath(arguments[0])
    if os.name == "nt" and not engine.lower().endswith(".exe") and os.path.exists(engine + ".exe"):
        engine += ".exe"
    gate = Gate(engine)
    os.makedirs(os.path.join(ROOT, "build"), exist_ok=True)
    scratch_absolute = tempfile.mkdtemp(prefix="uci-check-", dir=os.path.join(ROOT, "build"))
    scratch = os.path.relpath(scratch_absolute, ROOT).replace(os.sep, "/")
    start = time.monotonic()
    try:
        check_notation(gate)
        nnue_paths = check_nnue(gate, scratch)
        check_cli_misc(gate)
        check_options(gate, nnue_paths)
        check_books(gate, scratch)
        check_positions_and_go(gate)
        check_malformed(gate)
        if is_terminal:
            check_terminal(gate)
    finally:
        shutil.rmtree(scratch_absolute, ignore_errors=True)
    total = gate.passed + gate.failed
    print(f"{'[PASS]' if gate.failed == 0 else '[FAIL]'} {'uci':<6} {f'{gate.passed}/{total}':<9} {'':>21} "
          f"{'':<25} {time.monotonic() - start:8.3f}s {'':>11} (UCI protocol and subcommands, through pipes)",
          flush=True)
    return 0 if gate.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
