#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonny Reckless
"""Generate the engine's compile-time constant tables (src/*.inc).

Emits:
  src/zobrist_tables.inc — the Zobrist keys, drawn from the xorshift64* PRNG in the engine's historical
                           order (seed 0x9E3779B97F4A7C15). The draw ORDER is part of the engine's identity:
                           any change shifts every position key, the TT behaviour, and the bench signature.
  src/eval_tables.inc    — the PeSTO piece-square tables combined with material values, exactly as the old
                           init_eval() computed them at startup.
  src/ln_tables.inc      — ln(n) for n = 1..127 in Q28 fixed point (round(ln(n) * 2^28)), for the LMR
                           reduction table: the search then needs no floating point at all, so the bench
                           node signature is reproducible on every platform/compiler by construction.
  src/bitboard_tables.inc— the leaper attack tables (pawn/knight/king) and the BetweenBB/LineBB geometry
                           tables: pure functions of square geometry, mirrored exactly from the historical
                           init_bitboards() construction (only the magic sliding-attack tables remain
                           runtime-built).

The tables are true constants, so generating them once (and committing the output) makes them `static const`
— thread-safe by construction, no init-order dependency — instead of write-once globals. Regenerate only if
the PST data or the Zobrist scheme deliberately changes; the bench node signature will catch any accidental
difference (it depends on every one of these values).

Usage: python3 tools/generate_tables.py   (writes the two .inc files in place)
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
MASK = (1 << 64) - 1

# ---- Zobrist: xorshift64* in the exact historical draw order (see position.c's init_zobrist history) ----


def zobrist_stream(seed):
    state = seed
    while True:
        state ^= (state >> 12)
        state ^= (state << 25) & MASK
        state ^= (state >> 27)
        yield (state * 0x2545F4914F6CDD1D) & MASK


def generate_zobrist():
    draw = zobrist_stream(0x9E3779B97F4A7C15)
    piece = [[[0] * 64 for _ in range(7)] for _ in range(2)]  # [color][piece 0..6][square]; row 0 unused
    for code in range(12):
        color, ptype = code // 6, code % 6 + 1
        for square in range(64):
            piece[color][ptype][square] = next(draw)
    castle = [next(draw) for _ in range(16)]
    ep = [0] * 64
    for file in range(8):
        key = next(draw)
        ep[2 * 8 + file] = key  # rank 3 (White double push)
        ep[5 * 8 + file] = key  # rank 6 (Black double push)
    side = next(draw)
    return piece, castle, ep, side


# ---- PeSTO source data (single source of truth; the .inc files are the only C copies) ----

MG_VALUE = [82, 337, 365, 477, 1025, 0]
EG_VALUE = [94, 281, 297, 512, 936, 0]

# The published PeSTO piece-square tables (Pawn Advanced eval by Ronald Friederich), the single
# source of truth; eval.c holds only the generated combined tables.
PST = {
    "mg_pawn": [
        0, 0, 0, 0, 0, 0, 0, 0,
        98, 134, 61, 95, 68, 126, 34, -11,
        -6, 7, 26, 31, 65, 56, 25, -20,
        -14, 13, 6, 21, 23, 12, 17, -23,
        -27, -2, -5, 12, 17, 6, 10, -25,
        -26, -4, -4, -10, 3, 3, 33, -12,
        -35, -1, -20, -23, -15, 24, 38, -22,
        0, 0, 0, 0, 0, 0, 0, 0,
    ],
    "eg_pawn": [
        0, 0, 0, 0, 0, 0, 0, 0,
        178, 173, 158, 134, 147, 132, 165, 187,
        94, 100, 85, 67, 56, 53, 82, 84,
        32, 24, 13, 5, -2, 4, 17, 17,
        13, 9, -3, -7, -7, -8, 3, -1,
        4, 7, -6, 1, 0, -5, -1, -8,
        13, 8, 8, 10, 13, 0, 2, -7,
        0, 0, 0, 0, 0, 0, 0, 0,
    ],
    "mg_knight": [
        -167, -89, -34, -49, 61, -97, -15, -107,
        -73, -41, 72, 36, 23, 62, 7, -17,
        -47, 60, 37, 65, 84, 129, 73, 44,
        -9, 17, 19, 53, 37, 69, 18, 22,
        -13, 4, 16, 13, 28, 19, 21, -8,
        -23, -9, 12, 10, 19, 17, 25, -16,
        -29, -53, -12, -3, -1, 18, -14, -19,
        -105, -21, -58, -33, -17, -28, -19, -23,
    ],
    "eg_knight": [
        -58, -38, -13, -28, -31, -27, -63, -99,
        -25, -8, -25, -2, -9, -25, -24, -52,
        -24, -20, 10, 9, -1, -9, -19, -41,
        -17, 3, 22, 22, 22, 11, 8, -18,
        -18, -6, 16, 25, 16, 17, 4, -18,
        -23, -3, -1, 15, 10, -3, -20, -22,
        -42, -20, -10, -5, -2, -20, -23, -44,
        -29, -51, -23, -15, -22, -18, -50, -64,
    ],
    "mg_bishop": [
        -29, 4, -82, -37, -25, -42, 7, -8,
        -26, 16, -18, -13, 30, 59, 18, -47,
        -16, 37, 43, 40, 35, 50, 37, -2,
        -4, 5, 19, 50, 37, 37, 7, -2,
        -6, 13, 13, 26, 34, 12, 10, 4,
        0, 15, 15, 15, 14, 27, 18, 10,
        4, 15, 16, 0, 7, 21, 33, 1,
        -33, -3, -14, -21, -13, -12, -39, -21,
    ],
    "eg_bishop": [
        -14, -21, -11, -8, -7, -9, -17, -24,
        -8, -4, 7, -12, -3, -13, -4, -14,
        2, -8, 0, -1, -2, 6, 0, 4,
        -3, 9, 12, 9, 14, 10, 3, 2,
        -6, 3, 13, 19, 7, 10, -3, -9,
        -12, -3, 8, 10, 13, 3, -7, -15,
        -14, -18, -7, -1, 4, -9, -15, -27,
        -23, -9, -23, -5, -9, -16, -5, -17,
    ],
    "mg_rook": [
        32, 42, 32, 51, 63, 9, 31, 43,
        27, 32, 58, 62, 80, 67, 26, 44,
        -5, 19, 26, 36, 17, 45, 61, 16,
        -24, -11, 7, 26, 24, 35, -8, -20,
        -36, -26, -12, -1, 9, -7, 6, -23,
        -45, -25, -16, -17, 3, 0, -5, -33,
        -44, -16, -20, -9, -1, 11, -6, -71,
        -19, -13, 1, 17, 16, 7, -37, -26,
    ],
    "eg_rook": [
        13, 10, 18, 15, 12, 12, 8, 5,
        11, 13, 13, 11, -3, 3, 8, 3,
        7, 7, 7, 5, 4, -3, -5, -3,
        4, 3, 13, 1, 2, 1, -1, 2,
        3, 5, 8, 4, -5, -6, -8, -11,
        -4, 0, -5, -1, -7, -12, -8, -16,
        -6, -6, 0, 2, -9, -9, -11, -3,
        -9, 2, 3, -1, -5, -13, 4, -20,
    ],
    "mg_queen": [
        -28, 0, 29, 12, 59, 44, 43, 45,
        -24, -39, -5, 1, -16, 57, 28, 54,
        -13, -17, 7, 8, 29, 56, 47, 57,
        -27, -27, -16, -16, -1, 17, -2, 1,
        -9, -26, -9, -10, -2, -4, 3, -3,
        -14, 2, -11, -2, -5, 2, 14, 5,
        -35, -8, 11, 2, 8, 15, -3, 1,
        -1, -18, -9, 10, -15, -25, -31, -50,
    ],
    "eg_queen": [
        -9, 22, 22, 27, 27, 19, 10, 20,
        -17, 20, 32, 41, 58, 25, 30, 0,
        -20, 6, 9, 49, 47, 35, 19, 9,
        3, 22, 24, 45, 57, 40, 57, 36,
        -18, 28, 19, 47, 31, 34, 39, 23,
        -16, -27, 15, 6, 9, 17, 10, 5,
        -22, -23, -30, -16, -16, -23, -36, -32,
        -33, -28, -22, -43, -5, -32, -20, -41,
    ],
    "mg_king": [
        -65, 23, 16, -15, -56, -34, 2, 13,
        29, -1, -20, -7, -8, -4, -38, -29,
        -9, 24, 2, -16, -20, 6, 22, -22,
        -17, -20, -12, -27, -30, -25, -14, -36,
        -49, -1, -27, -39, -46, -44, -33, -51,
        -14, -14, -22, -46, -44, -30, -15, -27,
        1, 7, -8, -64, -43, -16, 9, 8,
        -15, 36, 12, -54, 8, -28, 24, 14,
    ],
    "eg_king": [
        -74, -35, -18, -18, -11, 15, 4, -17,
        -12, 17, 14, 17, 17, 38, 23, 11,
        10, 17, 23, 15, 20, 45, 44, 13,
        -8, 22, 24, 27, 26, 33, 26, 3,
        -18, -4, 21, 24, 27, 23, 9, -11,
        -19, -3, 11, 21, 23, 16, 7, -9,
        -27, -11, 4, 13, 14, 4, -5, -17,
        -53, -34, -21, -11, -28, -14, -24, -43,
    ],
}


def generate_pesto(pst):
    order = ["pawn", "knight", "bishop", "rook", "queen", "king"]
    mg = [[0] * 64 for _ in range(12)]
    eg = [[0] * 64 for _ in range(12)]
    for piece in range(6):
        mg_pst, eg_pst = pst["mg_" + order[piece]], pst["eg_" + order[piece]]
        for square in range(64):
            mg[piece][square] = MG_VALUE[piece] + mg_pst[square ^ 56]  # white
            eg[piece][square] = EG_VALUE[piece] + eg_pst[square ^ 56]
            mg[piece + 6][square] = MG_VALUE[piece] + mg_pst[square]  # black
            eg[piece + 6][square] = EG_VALUE[piece] + eg_pst[square]
    return mg, eg


# ---- emission --------------------------------------------------------------------------------------------

HEADER = """// GENERATED by tools/generate_tables.py — DO NOT EDIT.
// Regenerate only on a deliberate change to the underlying data; the bench node signature depends on every
// value here and will catch any accidental difference.
"""


def fmt_u64_rows(values, per_row=4):
    rows = []
    for i in range(0, len(values), per_row):
        rows.append("    " + ", ".join(f"0x{v:016x}ULL" for v in values[i:i + per_row]) + ",")
    return "\n".join(rows)


def fmt_int_rows(values, per_row=12):
    rows = []
    for i in range(0, len(values), per_row):
        rows.append("        " + ", ".join(str(v) for v in values[i:i + per_row]) + ",")
    return "\n".join(rows)


def emit_zobrist(piece, castle, ep, side):
    out = [HEADER]
    out.append("// clang-format off")
    out.append("static const uint64_t ZobristPiece[NUM_COLORS][NUM_PIECES][NUM_SQUARES] = {")
    for color in range(2):
        out.append("    {")
        for ptype in range(7):
            out.append("        {")
            vals = piece[color][ptype]
            for i in range(0, 64, 4):
                out.append("            " + ", ".join(f"0x{v:016x}ULL" for v in vals[i:i + 4]) + ",")
            out.append("        },")
        out.append("    },")
    out.append("};")
    out.append("")
    out.append("static const uint64_t ZobristCastle[16] = {")
    out.append(fmt_u64_rows(castle))
    out.append("};")
    out.append("")
    out.append("static const uint64_t ZobristEp[NUM_SQUARES] = {")
    out.append(fmt_u64_rows(ep))
    out.append("};")
    out.append("")
    out.append(f"static const uint64_t ZobristSide = 0x{side:016x}ULL;")
    out.append("// clang-format on")
    out.append("")
    (ROOT / "src" / "zobrist_tables.inc").write_text("\n".join(out))


def emit_pesto(mg, eg):
    out = [HEADER]
    out.append("// clang-format off")
    for name, table in (("mg_table", mg), ("eg_table", eg)):
        out.append(f"static const int {name}[12][64] = {{")
        for row in table:
            out.append("    {")
            out.append(fmt_int_rows(row))
            out.append("    },")
        out.append("};")
        out.append("")
    out.append("// clang-format on")
    out.append("")
    (ROOT / "src" / "eval_tables.inc").write_text("\n".join(out))


# ---- bitboard geometry tables (leapers + between/line), mirroring the historical init_bitboards ----

FULL = (1 << 64) - 1


def sq_bb(square):
    return 1 << square


def shift(bb, delta, avoid_file_mask):
    bb &= ~avoid_file_mask & FULL
    return ((bb << delta) if delta > 0 else (bb >> -delta)) & FULL


FILE_A = sum(1 << (8 * r) for r in range(8))
FILE_H = FILE_A << 7


def sliding_attack(square, occupancy, deltas):
    """Exact mirror of bitboard.c's ray walk (file-wrap rejected by per-step file distance)."""
    attacks = 0
    for delta in deltas:
        current = square
        while True:
            previous_file = current & 7
            nxt = current + delta
            if nxt < 0 or nxt >= 64 or abs((nxt & 7) - previous_file) > 1:
                break
            attacks |= sq_bb(nxt)
            if occupancy & sq_bb(nxt):
                break
            current = nxt
    return attacks


ROOK_DIRS = [8, -8, 1, -1]
BISHOP_DIRS = [9, 7, -7, -9]


def generate_bitboard_tables():
    pawn = [[0] * 64 for _ in range(2)]
    knight = [0] * 64
    king = [0] * 64
    for square in range(64):
        bb = sq_bb(square)
        pawn[0][square] = (shift(bb, 9, FILE_H) | shift(bb, 7, FILE_A))    # white: NE | NW
        pawn[1][square] = (shift(bb, -7, FILE_H) | shift(bb, -9, FILE_A))  # black: SE | SW
        file, rank = square & 7, square >> 3
        for df, dr in [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]:
            if 0 <= file + df < 8 and 0 <= rank + dr < 8:
                knight[square] |= sq_bb((rank + dr) * 8 + file + df)
        for df, dr in [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]:
            if 0 <= file + df < 8 and 0 <= rank + dr < 8:
                king[square] |= sq_bb((rank + dr) * 8 + file + df)

    between = [[0] * 64 for _ in range(64)]
    line = [[0] * 64 for _ in range(64)]
    for frm in range(64):
        for to in range(64):
            if frm == to:
                continue
            for dirs in (ROOK_DIRS, BISHOP_DIRS):
                if sliding_attack(frm, 0, dirs) & sq_bb(to):
                    between[frm][to] = sliding_attack(frm, sq_bb(to), dirs) & sliding_attack(to, sq_bb(frm), dirs)
                    line[frm][to] = ((sq_bb(frm) | sliding_attack(frm, 0, dirs)) &
                                     (sq_bb(to) | sliding_attack(to, 0, dirs)))
    return pawn, knight, king, between, line


def emit_bitboard_tables():
    pawn, knight, king, between, line = generate_bitboard_tables()
    out = [HEADER]
    out.append("// clang-format off")

    def emit_1d(name, values, dim):
        out.append(f"const Bitboard {name}[{dim}] = {{")
        out.append(fmt_u64_rows(values))
        out.append("};")
        out.append("")

    out.append("const Bitboard PawnAttacks[NUM_COLORS][64] = {")
    for color in range(2):
        out.append("    {")
        for i in range(0, 64, 4):
            out.append("        " + ", ".join(f"0x{v:016x}ULL" for v in pawn[color][i:i + 4]) + ",")
        out.append("    },")
    out.append("};")
    out.append("")
    emit_1d("KnightAttacks", knight, 64)
    emit_1d("KingAttacks", king, 64)
    for name, table in (("BetweenBB", between), ("LineBB", line)):
        out.append(f"const Bitboard {name}[64][64] = {{")
        for row in table:
            out.append("    {")
            for i in range(0, 64, 4):
                out.append("        " + ", ".join(f"0x{v:016x}ULL" for v in row[i:i + 4]) + ",")
            out.append("    },")
        out.append("};")
        out.append("")
    out.append("// clang-format on")
    out.append("")
    (ROOT / "src" / "bitboard_tables.inc").write_text("\n".join(out))


def emit_ln():
    import math
    out = [HEADER]
    out.append("// ln(n) in Q28 fixed point (round(ln(n) * 2^28)); index 0 unused. int64_t so products stay 64-bit.")
    out.append("// clang-format off")
    out.append("static const int64_t LnQ28[128] = {")
    values = [0] + [round(math.log(n) * (1 << 28)) for n in range(1, 128)]
    for i in range(0, 128, 8):
        out.append("    " + ", ".join(str(v) for v in values[i:i + 8]) + ",")
    out.append("};")
    out.append("// clang-format on")
    out.append("")
    (ROOT / "src" / "ln_tables.inc").write_text("\n".join(out))


def main():
    piece, castle, ep, side = generate_zobrist()
    emit_zobrist(piece, castle, ep, side)
    mg, eg = generate_pesto(PST)
    emit_pesto(mg, eg)
    emit_ln()
    emit_bitboard_tables()
    print("wrote src/zobrist_tables.inc, src/eval_tables.inc, src/ln_tables.inc, and src/bitboard_tables.inc")


if __name__ == "__main__":
    main()
