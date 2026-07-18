#include "movegen.h"
#include "bitboard.h"

namespace
{

void add_promotions(MoveList &l, int from, int to, bool cap)
{
    uint16_t base = cap ? FLAG_PROMO_CAP_N : FLAG_PROMO_N;
    // Queen first (best for move ordering), then knight/rook/bishop.
    l.add(Move(from, to, base + (QUEEN - KNIGHT)));
    l.add(Move(from, to, base + (KNIGHT - KNIGHT)));
    l.add(Move(from, to, base + (ROOK - KNIGHT)));
    l.add(Move(from, to, base + (BISHOP - KNIGHT)));
}

void gen_pseudo(const Position &pos, MoveList &l, bool noisy)
{
    const Color    us = pos.stm, them = ~us;
    const Bitboard occ   = pos.occupied();
    const Bitboard own   = pos.byColor[us];
    const Bitboard enemy = pos.byColor[them];
    const Bitboard empty = ~occ;

    // --- Pawns (per-pawn for clarity; correctness before speed) ---
    Bitboard pawns = pos.pieces(us, PAWN);
    int      up    = us == WHITE ? 8 : -8;
    while (pawns)
    {
        int  sq        = pop_lsb(pawns);
        int  one       = sq + up;
        bool promoRank = relative_rank(us, one) == 7;

        // Captures + promotions on capture.
        Bitboard caps = pawn_attacks(us, sq) & enemy;
        while (caps)
        {
            int to = pop_lsb(caps);
            if (promoRank)
            {
                add_promotions(l, sq, to, true);
            }
            else
            {
                l.add(Move(sq, to, FLAG_CAPTURE));
            }
        }
        // En passant.
        if (pos.epSq != NO_SQ && (pawn_attacks(us, sq) & sq_bb(pos.epSq)))
        {
            l.add(Move(sq, pos.epSq, FLAG_EP));
        }

        // Quiet pushes (skipped when generating noisy-only, except quiet promotions which are noisy).
        if (empty & sq_bb(one))
        {
            if (promoRank)
            {
                add_promotions(l, sq, one, false);
            }
            else if (!noisy)
            {
                l.add(Move(sq, one, FLAG_QUIET));
                if (relative_rank(us, sq) == 1 && (empty & sq_bb(one + up)))
                {
                    l.add(Move(sq, one + up, FLAG_DOUBLE));
                }
            }
        }
    }

    // --- Knights / King (leapers) ---
    Bitboard knights = pos.pieces(us, KNIGHT);
    while (knights)
    {
        int      sq = pop_lsb(knights);
        Bitboard t  = knight_attacks(sq) & ~own;
        if (noisy)
        {
            t &= enemy;
        }
        while (t)
        {
            int to = pop_lsb(t);
            l.add(Move(sq, to, (enemy & sq_bb(to)) ? FLAG_CAPTURE : FLAG_QUIET));
        }
    }
    int ksq = pos.king_sq(us);
    {
        Bitboard t = king_attacks(ksq) & ~own;
        if (noisy)
        {
            t &= enemy;
        }
        while (t)
        {
            int to = pop_lsb(t);
            l.add(Move(ksq, to, (enemy & sq_bb(to)) ? FLAG_CAPTURE : FLAG_QUIET));
        }
    }

    // --- Sliders ---
    auto sliders = [&](Bitboard bb, auto attack) {
        while (bb)
        {
            int      sq = pop_lsb(bb);
            Bitboard t  = attack(sq, occ) & ~own;
            if (noisy)
            {
                t &= enemy;
            }
            while (t)
            {
                int to = pop_lsb(t);
                l.add(Move(sq, to, (enemy & sq_bb(to)) ? FLAG_CAPTURE : FLAG_QUIET));
            }
        }
    };
    sliders(pos.pieces(us, BISHOP), bishop_attacks);
    sliders(pos.pieces(us, ROOK), rook_attacks);
    sliders(pos.pieces(us, QUEEN), queen_attacks);

    // --- Castling (fully legal: not in check, path empty and unattacked) ---
    if (!noisy && !pos.attacked_by(ksq, them))
    {
        int     r     = us == WHITE ? 0 : 7;
        uint8_t kFlag = us == WHITE ? CR_WK : CR_BK;
        uint8_t qFlag = us == WHITE ? CR_WQ : CR_BQ;
        int     e = make_square(4, r), f = make_square(5, r), g = make_square(6, r);
        int     d = make_square(3, r), c = make_square(2, r), b = make_square(1, r);
        if ((pos.castling & kFlag) && (empty & sq_bb(f)) && (empty & sq_bb(g)) && !pos.attacked_by(f, them) &&
            !pos.attacked_by(g, them))
        {
            l.add(Move(e, g, FLAG_KCASTLE));
        }
        if ((pos.castling & qFlag) && (empty & sq_bb(d)) && (empty & sq_bb(c)) && (empty & sq_bb(b)) &&
            !pos.attacked_by(d, them) && !pos.attacked_by(c, them))
        {
            l.add(Move(e, c, FLAG_QCASTLE));
        }
    }
}

} // namespace

void generate_legal(const Position &pos, MoveList &list, bool noisyOnly)
{
    MoveList pseudo;
    gen_pseudo(pos, pseudo, noisyOnly);
    for (Move m : pseudo)
    {
        if (pos.is_legal(m))
        {
            list.add(m);
        }
    }
}
