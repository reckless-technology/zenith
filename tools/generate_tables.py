#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonny Reckless
"""Generate the engine's compile-time constant tables (src/generated/*.inc).

Emits:
  src/generated/zobrist_tables.inc — the Zobrist keys, drawn from the xorshift64* PRNG in the engine's historical
                           order (seed 0x9E3779B97F4A7C15). The draw ORDER is part of the engine's identity:
                           any change shifts every position key, the TT behaviour, and the bench signature.
  src/generated/eval_tables.inc    — the PeSTO piece-square tables combined with material values, exactly as the old
                           init_eval() computed them at startup.
  src/generated/ln_tables.inc      — ln(n) for n = 1..127 in Q28 fixed point (round(ln(n) * 2^28)), for the LMR
                           reduction table: the search then needs no floating point at all, so the bench
                           node signature is reproducible on every platform/compiler by construction.
  src/generated/bitboard_tables.inc— the leaper attack tables (pawn/knight/king) and the BetweenBB/LineBB geometry
                           tables: pure functions of square geometry, mirrored exactly from the historical
                           init_bitboards() construction.
  src/generated/magic_tables.inc   — the 64 rook + 64 bishop magic multipliers, found by replicating the engine's
                           historical startup search bit-for-bit (same xorshift64* seeds, same sparse
                           candidates, same acceptance test), so startup just fills the attack tables with
                           known-good magics instead of searching (only that fill remains runtime work).

The tables are true constants, so generating them once (and committing the output) makes them `static const`
— thread-safe by construction, no init-order dependency — instead of write-once globals. Regenerate only if
the PST data or the Zobrist scheme deliberately changes; the bench node signature will catch any accidental
difference (it depends on every one of these values).

Usage: python3 tools/generate_tables.py   (writes the five .inc files in place)
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent  # repository root (this file lives in tools/)
MASK = (1 << 64) - 1  # Python ints are unbounded; AND with this after every op that must wrap like a C uint64_t

# ---- Zobrist: xorshift64* in the exact historical draw order (see position.c's init_zobrist history) ----


def zobrist_stream(seed):
    """Yield the engine's Zobrist key stream: xorshift64* with the historical constants.

    This mirrors the deleted init_zobrist()'s PRNG exactly (shifts 12/25/27, multiplier
    0x2545F4914F6CDD1D). The internal state advances by the three xorshifts; each yielded key is the
    state times the multiplier, truncated to 64 bits. Both the seed and the DRAW ORDER are part of the
    engine's identity: any change shifts every position key, and with them the TT behavior and the bench
    node signature.
    """
    state = seed
    while True:
        state ^= (state >> 12)
        state ^= (state << 25) & MASK
        state ^= (state >> 27)
        yield (state * 0x2545F4914F6CDD1D) & MASK


def generate_zobrist():
    """Draw every Zobrist key in the historical order; return (piece, castle, ep, side) tables.

    Order (fixed forever): 12 (color, piece-type) blocks of 64 squares — white pawn..king, then black —
    then the 16 castling-rights masks, then 8 en-passant file keys (each replicated onto that file's two
    possible ep target squares: rank 3 for a White double push, rank 6 for Black), then the side-to-move
    key. The piece table is shaped [color][piece 0..6][square] to match the engine's Piece enum
    (NO_PIECE = 0 row left zero and unused).
    """
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
    """Combine material values with the PST data into the engine's 12x64 tapered-eval tables.

    Exactly what the deleted init_eval() computed at startup: row layout is white piece types 0..5 then
    black 6..11 (0-based types); the published PSTs are printed with index 0 = a8, so WHITE reads
    pst[square ^ 56] (vertical flip) and BLACK reads pst[square] directly. Returns (mg, eg) tables of
    middlegame and endgame centipawn values.
    """
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
    """Format a flat list of ints as indented C rows of 0x...ULL literals, per_row values per line."""
    rows = []
    for i in range(0, len(values), per_row):
        rows.append("    " + ", ".join(f"0x{v:016x}ULL" for v in values[i:i + per_row]) + ",")
    return "\n".join(rows)


def fmt_int_rows(values, per_row=12):
    """Format a flat list of ints as deeper-indented C rows of decimal literals (for the PST tables)."""
    rows = []
    for i in range(0, len(values), per_row):
        rows.append("        " + ", ".join(str(v) for v in values[i:i + per_row]) + ",")
    return "\n".join(rows)


def emit_zobrist(piece, castle, ep, side):
    """Write zobrist_tables.inc: the four static const Zobrist tables position.c includes."""
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
    (ROOT / "src" / "generated" / "zobrist_tables.inc").write_text("\n".join(out))


def emit_pesto(mg, eg):
    """Write eval_tables.inc: the combined material+PST mg/eg tables eval.c includes."""
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
    (ROOT / "src" / "generated" / "eval_tables.inc").write_text("\n".join(out))


# ---- bitboard geometry tables (leapers + between/line), mirroring the historical init_bitboards ----

FULL = (1 << 64) - 1  # same 64-bit wrap mask as MASK, named for the bitboard sections below


def sq_bb(square):
    """Bitboard with only @square set (A1 = 0 ... H8 = 63, rank-major like the engine)."""
    return 1 << square


def shift(bb, delta, avoid_file_mask):
    """Shift a bitboard by a square delta, first clearing @avoid_file_mask to stop file wraps.

    Mirrors the engine's shift_northeast/... helpers: e.g. a NE shift (+9) must clear FILE_H first or an
    h-file bit would wrap onto the a-file of the next rank.
    """
    bb &= ~avoid_file_mask & FULL
    return ((bb << delta) if delta > 0 else (bb >> -delta)) & FULL


FILE_A = sum(1 << (8 * r) for r in range(8))
FILE_H = FILE_A << 7


def sliding_attack(square, occupancy, deltas):
    """Ray-walk sliding attacks from @square over @occupancy — an exact mirror of bitboard.c's version.

    Walks each direction delta square by square, including every square reached; a blocker square is
    included and then stops the ray. Board-edge wraps are rejected the same way as the C code: any single
    king-step move changes the file by at most 1, so a step whose file distance exceeds 1 ran off the
    east/west edge. Used for the leaper-free geometry tables (Between/Line), the magic masks, and the
    reference attack sets the magic search validates against.
    """
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


ROOK_DIRS = [8, -8, 1, -1]    # N, S, E, W as square deltas (matches bitboard.c's RookDirs)
BISHOP_DIRS = [9, 7, -7, -9]  # NE, NW, SE, SW (matches BishopDirs)


def generate_bitboard_tables():
    """Build the five geometry tables exactly as the deleted init_bitboards() leaper/line code did.

    - PawnAttacks[color][sq]: the one or two capture squares (NE|NW for White, SE|SW for Black).
    - Knight/KingAttacks[sq]: offset tables with edge rejection by file/rank bounds.
    - BetweenBB[a][b]: squares strictly between two aligned squares (exclusive), else 0 — computed as the
      intersection of each endpoint's ray attacks with the other endpoint as the sole blocker.
    - LineBB[a][b]: the whole rank/file/diagonal through both squares (endpoints INCLUDED), else 0 —
      the intersection of the endpoints' empty-board rays, each with its own square OR-ed in.
    Returns (pawn, knight, king, between, line).
    """
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
    """Write bitboard_tables.inc: the extern const leaper/geometry tables bitboard.c defines."""
    pawn, knight, king, between, line = generate_bitboard_tables()
    out = [HEADER]
    out.append("// clang-format off")

    def emit_1d(name, values, dim):
        # one flat const Bitboard array (the knight/king tables)
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
    (ROOT / "src" / "generated" / "bitboard_tables.inc").write_text("\n".join(out))


# ---- magic multipliers: replicate the engine's historical startup search exactly ----

RANK_1 = 0xFF
RANK_8 = RANK_1 << 56


def rank_bb(square):
    """Bitboard of @square's whole rank (for the magic mask's edge trimming)."""
    return (RANK_1 << (8 * (square >> 3))) & FULL


def file_bb(square):
    """Bitboard of @square's whole file (for the magic mask's edge trimming)."""
    return (FILE_A << (square & 7)) & FULL


def prng_next(state):
    """One xorshift64* step — bit-for-bit the C prng_next the magic search used. Returns (state, draw)."""
    state ^= state >> 12
    state ^= (state << 25) & FULL
    state ^= state >> 27
    return state, (state * 0x2545F4914F6CDD1D) & FULL


def prng_sparse(state):
    """AND of three consecutive draws: few set bits, which makes good magic-multiplier candidates."""
    state, a = prng_next(state)
    state, b = prng_next(state)
    state, c = prng_next(state)
    return state, a & b & c


def find_magics(is_rook, deltas):
    """Find the 64 magic multipliers for one slider — a bit-for-bit mirror of init_magics' old search.

    Per square: the relevant-occupancy mask is the empty-board rays minus the edges not on the piece's own
    rank/file (an edge blocker never changes the attack set); every subset of the mask is enumerated with
    the Carry-Rippler trick alongside its true (ray-walked) attack set. Candidate multipliers come from the
    same seeded sparse PRNG stream the engine used (seed mixes the square and a rook/bishop tag), gated by
    the same quick quality filter (the top byte of mask*magic must have >= 6 bits set), and a candidate is
    accepted when mapping every subset through (subset*magic) >> (64-relevant_bits) produces no destructive
    collision (two subsets may share a slot only if their attack sets are identical). The epoch dict makes
    each attempt O(subsets) without clearing the table. Because seeds, stream, and acceptance are identical
    to the deleted C search, the emitted multipliers are exactly the ones the engine used to find at
    startup, so the filled attack tables are bit-identical too.
    """
    magics = []
    for square in range(64):
        edges = ((RANK_1 | RANK_8) & ~rank_bb(square)) | (((FILE_A | FILE_H) & FULL) & ~file_bb(square))
        mask = sliding_attack(square, 0, deltas) & ~edges
        relevant_bits = bin(mask).count("1")
        shift = 64 - relevant_bits

        subsets, references = [], []
        subset = 0
        while True:
            subsets.append(subset)
            references.append(sliding_attack(square, subset, deltas))
            subset = (subset - mask) & mask & FULL
            if subset == 0:
                break

        state = (0x9E3779B97F4A7C15 ^ ((square * 0xBF58476D1CE4E5B9) & FULL) ^ (1 if is_rook else 2)) & FULL
        epoch = {}
        epoch_counter = 0
        table = {}
        while True:
            magic = 0
            while bin(((mask * magic) & FULL) >> 56).count("1") < 6:
                state, magic = prng_sparse(state)
            epoch_counter += 1
            ok = True
            for occ, ref in zip(subsets, references):
                index = (((occ & mask) * magic & FULL) >> shift)
                if epoch.get(index, 0) < epoch_counter:
                    epoch[index] = epoch_counter
                    table[index] = ref
                elif table[index] != ref:
                    ok = False
                    break
            if ok:
                magics.append(magic)
                break
    return magics


def emit_magics():
    """Write magic_tables.inc: the rook/bishop multiplier arrays init_magics fills the tables from."""
    rook = find_magics(True, ROOK_DIRS)
    bishop = find_magics(False, BISHOP_DIRS)
    out = [HEADER]
    out.append("// The magic multipliers the engine's historical startup search would find (replicated bit-for-bit),")
    out.append("// so init_magics only fills the attack tables. Unused by the PEXT build.")
    out.append("// clang-format off")
    for name, values in (("RookMagicNumbers", rook), ("BishopMagicNumbers", bishop)):
        out.append(f"static const Bitboard {name}[64] = {{")
        out.append(fmt_u64_rows(values))
        out.append("};")
        out.append("")
    out.append("// clang-format on")
    out.append("")
    (ROOT / "src" / "generated" / "magic_tables.inc").write_text("\n".join(out))


def emit_ln():
    """Write ln_tables.inc: ln(n) for n = 1..127 in Q28 fixed point, for the integer-only LMR table.

    Q28 is the widest format whose ln*ln products still fit int64 (ln(127)^2 in Q56 uses 61 bits), and its
    ~4e-9 rounding error sits about four orders of magnitude inside the closest LMR truncation boundary, so
    the integer reduction table is identical to the old double computation. Index 0 is unused (ln undefined);
    the entries are plain decimals since they are int64_t, not bitboards.
    """
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
    (ROOT / "src" / "generated" / "ln_tables.inc").write_text("\n".join(out))


def main():
    """Generate and write all five .inc files (deterministic: reruns are byte-identical)."""
    piece, castle, ep, side = generate_zobrist()
    emit_zobrist(piece, castle, ep, side)
    mg, eg = generate_pesto(PST)
    emit_pesto(mg, eg)
    emit_ln()
    emit_bitboard_tables()
    emit_magics()
    print("wrote src/generated/{zobrist,eval,ln,bitboard,magic}_tables.inc")


if __name__ == "__main__":
    main()
