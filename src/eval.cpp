#include "eval.h"
#include "bitboard.h"

// PeSTO tapered evaluation (Rofchade's piece-square tables). Material + PST interpolated between a
// middlegame and endgame score by a phase count. Strong, compact, and self-contained; NNUE replaces the
// whole function later behind the evaluate() seam. A few cheap, uncontroversial terms (bishop pair,
// mobility, tempo) sit on top.

namespace {

const int mg_value[6] = {82, 337, 365, 477, 1025, 0};
const int eg_value[6] = {94, 281, 297, 512, 936, 0};
const int phase_inc[6] = {0, 1, 1, 2, 4, 0};

// Tables are printed with index 0 = a8. White reads pst[sq ^ 56]; black reads pst[sq].
const int mg_pawn[64] = {
    0, 0, 0, 0, 0, 0, 0, 0, 98, 134, 61, 95, 68, 126, 34, -11, -6, 7, 26, 31, 65, 56,
    25, -20, -14, 13, 6, 21, 23, 12, 17, -23, -27, -2, -5, 12, 17, 6, 10, -25, -26, -4,
    -4, -10, 3, 3, 33, -12, -35, -1, -20, -23, -15, 24, 38, -22, 0, 0, 0, 0, 0, 0, 0, 0};
const int eg_pawn[64] = {
    0, 0, 0, 0, 0, 0, 0, 0, 178, 173, 158, 134, 147, 132, 165, 187, 94, 100, 85, 67, 56, 53,
    82, 84, 32, 24, 13, 5, -2, 4, 17, 17, 13, 9, -3, -7, -7, -8, 3, -1, 4, 7, -6, 1, 0, -5,
    -1, -8, 13, 8, 8, 10, 13, 0, 2, -7, 0, 0, 0, 0, 0, 0, 0, 0};
const int mg_knight[64] = {
    -167, -89, -34, -49, 61, -97, -15, -107, -73, -41, 72, 36, 23, 62, 7, -17, -47, 60, 37, 65, 84,
    129, 73, 44, -9, 17, 19, 53, 37, 69, 18, 22, -13, 4, 16, 13, 28, 19, 21, -8, -23, -9, 12, 10,
    19, 17, 25, -16, -29, -53, -12, -3, -1, 18, -14, -19, -105, -21, -58, -33, -17, -28, -19, -23};
const int eg_knight[64] = {
    -58, -38, -13, -28, -31, -27, -63, -99, -25, -8, -25, -2, -9, -25, -24, -52, -24, -20, 10, 9, -1,
    -9, -19, -41, -17, 3, 22, 22, 22, 11, 8, -18, -18, -6, 16, 25, 16, 17, 4, -18, -23, -3, -1, 15,
    10, -3, -20, -22, -42, -20, -10, -5, -2, -20, -23, -44, -29, -51, -23, -15, -22, -18, -50, -64};
const int mg_bishop[64] = {
    -29, 4, -82, -37, -25, -42, 7, -8, -26, 16, -18, -13, 30, 59, 18, -47, -16, 37, 43, 40, 35, 50,
    37, -2, -4, 5, 19, 50, 37, 37, 7, -2, -6, 13, 13, 26, 34, 12, 10, 4, 0, 15, 15, 15, 14, 27, 18,
    10, 4, 15, 16, 0, 7, 21, 33, 1, -33, -3, -14, -21, -13, -12, -39, -21};
const int eg_bishop[64] = {
    -14, -21, -11, -8, -7, -9, -17, -24, -8, -4, 7, -12, -3, -13, -4, -14, 2, -8, 0, -1, -2, 6, 0, 4,
    -3, 9, 12, 9, 14, 10, 3, 2, -6, 3, 13, 19, 7, 10, -3, -9, -12, -3, 8, 10, 13, 3, -7, -15, -14,
    -18, -7, -1, 4, -9, -15, -27, -23, -9, -23, -5, -9, -16, -5, -17};
const int mg_rook[64] = {
    32, 42, 32, 51, 63, 9, 31, 43, 27, 32, 58, 62, 80, 67, 26, 44, -5, 19, 26, 36, 17, 45, 61, 16,
    -24, -11, 7, 26, 24, 35, -8, -20, -36, -26, -12, -1, 9, -7, 6, -23, -45, -25, -16, -17, 3, 0, -5,
    -33, -44, -16, -20, -9, -1, 11, -6, -71, -19, -13, 1, 17, 16, 7, -37, -26};
const int eg_rook[64] = {
    13, 10, 18, 15, 12, 12, 8, 5, 11, 13, 13, 11, -3, 3, 8, 3, 7, 7, 7, 5, 4, -3, -5, -3, 4, 3, 13,
    1, 2, 1, -1, 2, 3, 5, 8, 4, -5, -6, -8, -11, -4, 0, -5, -1, -7, -12, -8, -16, -6, -6, 0, 2, -9,
    -9, -11, -3, -9, 2, 3, -1, -5, -13, 4, -20};
const int mg_queen[64] = {
    -28, 0, 29, 12, 59, 44, 43, 45, -24, -39, -5, 1, -16, 57, 28, 54, -13, -17, 7, 8, 29, 56, 47, 57,
    -27, -27, -16, -16, -1, 17, -2, 1, -9, -26, -9, -10, -2, -4, 3, -3, -14, 2, -11, -2, -5, 2, 14, 5,
    -35, -8, 11, 2, 8, 15, -3, 1, -1, -18, -9, 10, -15, -25, -31, -50};
const int eg_queen[64] = {
    -9, 22, 22, 27, 27, 19, 10, 20, -17, 20, 32, 41, 58, 25, 30, 0, -20, 6, 9, 49, 47, 35, 19, 9, 3,
    22, 24, 45, 57, 40, 57, 36, -18, 28, 19, 47, 31, 34, 39, 23, -16, -27, 15, 6, 9, 17, 10, 5, -22,
    -23, -30, -16, -16, -23, -36, -32, -33, -28, -22, -43, -5, -32, -20, -41};
const int mg_king[64] = {
    -65, 23, 16, -15, -56, -34, 2, 13, 29, -1, -20, -7, -8, -4, -38, -29, -9, 24, 2, -16, -20, 6, 22,
    -22, -17, -20, -12, -27, -30, -25, -14, -36, -49, -1, -27, -39, -46, -44, -33, -51, -14, -14, -22,
    -46, -44, -30, -15, -27, 1, 7, -8, -64, -43, -16, 9, 8, -15, 36, 12, -54, 8, -28, 24, 14};
const int eg_king[64] = {
    -74, -35, -18, -18, -11, 15, 4, -17, -12, 17, 14, 17, 17, 38, 23, 11, 10, 17, 23, 15, 20, 45, 44,
    13, -8, 22, 24, 27, 26, 33, 26, 3, -18, -4, 21, 24, 27, 23, 9, -11, -19, -3, 11, 21, 23, 16, 7,
    -9, -27, -11, 4, 13, 14, 4, -5, -17, -53, -34, -21, -11, -28, -14, -24, -43};

const int* mg_pst[6] = {mg_pawn, mg_knight, mg_bishop, mg_rook, mg_queen, mg_king};
const int* eg_pst[6] = {eg_pawn, eg_knight, eg_bishop, eg_rook, eg_queen, eg_king};

int mg_table[12][64];
int eg_table[12][64];

} // namespace

void init_eval() {
    for (int pt = 0; pt < 6; pt++)
        for (int sq = 0; sq < 64; sq++) {
            mg_table[pt][sq] = mg_value[pt] + mg_pst[pt][sq ^ 56]; // white
            eg_table[pt][sq] = eg_value[pt] + eg_pst[pt][sq ^ 56];
            mg_table[pt + 6][sq] = mg_value[pt] + mg_pst[pt][sq]; // black
            eg_table[pt + 6][sq] = eg_value[pt] + eg_pst[pt][sq];
        }
}

int evaluate(const Position& pos) {
    int mg[2] = {0, 0}, eg[2] = {0, 0}, phase = 0;

    Bitboard occ = pos.occupied();
    Bitboard b = occ;
    while (b) {
        int sq = pop_lsb(b);
        Piece p = pos.board[sq];
        Color c = color_of(p);
        mg[c] += mg_table[p][sq];
        eg[c] += eg_table[p][sq];
        phase += phase_inc[type_of(p)];
    }

    // Cheap extra terms.
    for (Color c : {WHITE, BLACK}) {
        int sign = c == WHITE ? 1 : -1;
        // Bishop pair.
        if (popcount(pos.pieces(c, BISHOP)) >= 2) {
            mg[c] += 22;
            eg[c] += 40;
        }
        // Mobility (attacked squares not occupied by own pieces), small weights.
        Bitboard own = pos.byColor[c];
        auto mob = [&](Bitboard bb, auto atk, int wmg, int weg) {
            while (bb) {
                int s = pop_lsb(bb);
                int m = popcount(atk(s, occ) & ~own);
                mg[c] += wmg * (m - 4);
                eg[c] += weg * (m - 4);
            }
        };
        mob(pos.pieces(c, BISHOP), bishop_attacks, 4, 4);
        mob(pos.pieces(c, ROOK), rook_attacks, 2, 4);
        mob(pos.pieces(c, QUEEN), queen_attacks, 1, 2);
        (void)sign;
    }

    int mgScore = mg[WHITE] - mg[BLACK];
    int egScore = eg[WHITE] - eg[BLACK];
    if (phase > 24) phase = 24;
    int score = (mgScore * phase + egScore * (24 - phase)) / 24; // White-relative
    int stmScore = (pos.stm == WHITE ? score : -score);
    return stmScore + 10; // tempo: a small bonus for the side to move
}
