#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonny Reckless
"""Display a Polyglot opening book in human-readable form: one line of play per output line, in SAN.

Walks the book from the start position, following every stored move depth-first (highest weight
first), and prints each complete book line as numbered SAN. A position reached again through a
different move order is printed once in full; later arrivals end their line with "(transposes)".

All chess knowledge comes from the zenith binary: `polykey` locates positions in the book,
`moves`/`applymoves` validate and apply them, and `san` renders each finished line — the displayer
itself never interprets a position. Works on any Polyglot .bin (including `make get-book`'s).

Usage: python3 tools/show_book.py books/zenith-book-r1.bin [--weights] [--max-lines N]
"""
import argparse
import struct
import subprocess

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
# Polyglot encodes castling as king-takes-rook; UCI wants the king's real target square.
CASTLE_FROM_POLYGLOT = {"e1h1": "e1g1", "e1a1": "e1c1", "e8h8": "e8g8", "e8a8": "e8c8"}


def run_cli(binary, *args):
    result = subprocess.run([binary, *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"zenith {args[0]} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def load_book(path):
    """Parse the .bin into {key: [(uci_ish, weight)]} sorted by descending weight."""
    book = {}
    with open(path, "rb") as handle:
        data = handle.read()
    if len(data) % 16 != 0:
        raise SystemExit(f"{path} is not a Polyglot book (size not a multiple of 16)")
    for offset in range(0, len(data), 16):
        key, move, weight, _learn = struct.unpack_from(">QHHI", data, offset)
        to_file, to_rank = move & 7, (move >> 3) & 7
        from_file, from_rank = (move >> 6) & 7, (move >> 9) & 7
        promotion = "  nbrq"[(move >> 12) & 7].strip()
        uci = f"{chr(97 + from_file)}{1 + from_rank}{chr(97 + to_file)}{1 + to_rank}{promotion}"
        book.setdefault(key, []).append((uci, weight))
    for moves in book.values():
        moves.sort(key=lambda entry: -entry[1])
    return book


def resolve_move(book_uci, legal_moves):
    """Map a book move onto the position's legal moves (translating Polyglot castling encoding)."""
    if book_uci in legal_moves:
        return book_uci
    translated = CASTLE_FROM_POLYGLOT.get(book_uci)
    return translated if translated in legal_moves else None


def format_line(binary, uci_moves, weights, show_weights):
    """Render a root line as numbered SAN (via the engine), optionally annotating weights."""
    san_moves = run_cli(binary, "san", START_FEN, *uci_moves).split()
    parts = []
    for index, san in enumerate(san_moves):
        if index % 2 == 0:
            parts.append(f"{index // 2 + 1}.")
        note = f"{{{weights[index]}}}" if show_weights else ""
        parts.append(san + note)
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("book", help="Polyglot .bin file")
    parser.add_argument("--engine", default="./build/zenith")
    parser.add_argument("--weights", action="store_true", help="annotate each move with its book weight")
    parser.add_argument("--max-lines", type=int, default=5000)
    args = parser.parse_args()

    book = load_book(args.book)
    printed_lines = 0
    seen_keys = set()

    def walk(fen, line_moves, line_weights):
        nonlocal printed_lines
        if printed_lines >= args.max_lines:
            return
        key = int(run_cli(args.engine, "polykey", fen), 16)
        entries = book.get(key, [])
        is_transposition = key in seen_keys and entries
        seen_keys.add(key)
        if not entries or is_transposition:
            if line_moves:
                suffix = "  (transposes)" if is_transposition else ""
                print(format_line(args.engine, line_moves, line_weights, args.weights) + suffix)
                printed_lines += 1
            return
        legal_moves = run_cli(args.engine, "moves", fen).split()
        for book_uci, weight in entries:
            uci = resolve_move(book_uci, legal_moves)
            if uci is None:
                print(f"# WARNING: book move {book_uci} is not legal in {fen}")
                continue
            child_fen = run_cli(args.engine, "applymoves", fen, uci)
            walk(child_fen, line_moves + [uci], line_weights + [weight])

    walk(START_FEN, [], [])
    total_entries = sum(len(moves) for moves in book.values())
    print(f"# {args.book}: {len(book)} positions, {total_entries} moves, {printed_lines} lines shown")


if __name__ == "__main__":
    main()
