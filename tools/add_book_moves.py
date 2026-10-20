#!/usr/bin/env python3
"""Add moves to a Polyglot book by hand: a new .bin with the given moves among the old book's entries.

    python3 tools/add_book_moves.py <in.bin> <out.bin> [--engine ./build/zenith] --add <fen|startpos> <uci> <weight> ...

Each --add names a position (a FEN, or `startpos`), a legal move in it in UCI and the weight to store it with. The
book's own entries are kept as they are; an added move the book already holds for that position is refused, so a
weight is never silently doubled. The output is key-sorted and big-endian with castling as king-takes-rook, as
tools/build_book.py emits (the same encoding, imported from it). As there, all chess knowledge comes from the engine:
`polykey` gives each position's key and `moves` holds the move to the legal ones.

books/zenith-book-r5.bin is books/zenith-book-r4.bin with 1.d4 added at the root (Jonny, 2026-10-20):

    python3 tools/add_book_moves.py books/zenith-book-r4.bin books/zenith-book-r5.bin --add startpos d2d4 42

r4 opens 1.e4 (115) and 1.c4 (54) and holds no 1.d4 there - the builder scored it outside its keep margin - but it
holds ~2,100 positions of lines after 1.d4, built as replies for the side facing it. Weight 42 makes 1.d4 42 / 211 =
19.9 % of the first moves and opens those lines.
"""
import argparse
import os
import struct
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_book import encode_move, is_castling_move  # noqa: E402

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
ENTRY = struct.Struct(">QHHI")  # key, move, weight, learn


def run_cli(binary, *args):
    """One engine subcommand's output."""
    result = subprocess.run([binary, *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"zenith {args[0]} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("book", help="the Polyglot .bin to start from")
    parser.add_argument("out", help="the .bin to write")
    parser.add_argument("--engine", default="./build/zenith")
    parser.add_argument("--add", nargs=3, action="append", required=True, metavar=("POSITION", "UCI", "WEIGHT"),
                        help="a position (FEN or startpos), a legal move in it, and its weight (1-65535)")
    args = parser.parse_args()
    engine = os.path.abspath(args.engine) if os.sep in args.engine or "/" in args.engine else args.engine

    with open(args.book, "rb") as handle:
        data = handle.read()
    if len(data) % ENTRY.size:
        raise SystemExit(f"{args.book}: {len(data)} bytes is not a whole number of {ENTRY.size}-byte entries")
    entries = [ENTRY.unpack_from(data, offset) for offset in range(0, len(data), ENTRY.size)]
    held = {(key, move) for key, move, _, _ in entries}

    for position, uci, weight_text in args.add:
        fen = START_FEN if position == "startpos" else position
        weight = int(weight_text)
        if not 1 <= weight <= 65535:
            raise SystemExit(f"weight {weight} is outside 1-65535")
        if uci not in run_cli(engine, "moves", fen).split():
            raise SystemExit(f"{uci} is not a legal move in {fen}")
        key = int(run_cli(engine, "polykey", fen), 16)
        move = encode_move(uci, is_castling_move(fen, uci))
        if (key, move) in held:
            raise SystemExit(f"{args.book} already holds {uci} in {fen}")
        held.add((key, move))
        entries.append((key, move, weight, 0))
        print(f"added {uci} at weight {weight} in {fen} (key {key:016x})")

    entries.sort()
    with open(args.out, "wb") as handle:
        for entry in entries:
            handle.write(ENTRY.pack(*entry))
    print(f"wrote {len(entries)} entries ({os.path.getsize(args.out):,} bytes) -> {args.out}")


if __name__ == "__main__":
    main()
