#include "movegen.h"
#include "bitboard.h"

namespace
{

void add_promotions(MoveList &list, int from, int to, bool capture)
{
    uint16_t base = capture ? FLAG_PROMO_CAP_N : FLAG_PROMO_N;
    // Queen first (best for move ordering), then knight/rook/bishop.
    list.add(Move(from, to, base + (QUEEN - KNIGHT)));
    list.add(Move(from, to, base + (KNIGHT - KNIGHT)));
    list.add(Move(from, to, base + (ROOK - KNIGHT)));
    list.add(Move(from, to, base + (BISHOP - KNIGHT)));
}

} // namespace

// Pseudo-legal moves. Castling is emitted fully legal (king not in/through check); every other move is
// legal iff it does not leave the mover's own king in check — the search tests that with a single
// make_move (see negamax/qsearch), avoiding the separate copy-make that generate_legal does per move.
void generate_pseudo(const Position &pos, MoveList &list, bool noisy)
{
    const Color    side = pos.stm, opponent = ~side;
    const Bitboard occupancy = pos.occupied();
    const Bitboard own       = pos.by_color[side];
    const Bitboard enemy     = pos.by_color[opponent];
    const Bitboard empty     = ~occupancy;

    // --- Pawns (per-pawn for clarity; correctness before speed) ---
    Bitboard pawns   = pos.pieces(side, PAWN);
    int      forward = side == WHITE ? 8 : -8;
    while (pawns)
    {
        int  square     = pop_lsb(pawns);
        int  one_step   = square + forward;
        bool promo_rank = relative_rank(side, one_step) == 7;

        // Captures + promotions on capture.
        Bitboard captures = pawn_attacks(side, square) & enemy;
        while (captures)
        {
            int to = pop_lsb(captures);
            if (promo_rank)
            {
                add_promotions(list, square, to, true);
            }
            else
            {
                list.add(Move(square, to, FLAG_CAPTURE));
            }
        }
        // En passant.
        if (pos.ep_sq != NO_SQ && (pawn_attacks(side, square) & sq_bb(pos.ep_sq)))
        {
            list.add(Move(square, pos.ep_sq, FLAG_EP));
        }

        // Quiet pushes (skipped when generating noisy-only, except quiet promotions which are noisy).
        if (empty & sq_bb(one_step))
        {
            if (promo_rank)
            {
                add_promotions(list, square, one_step, false);
            }
            else if (!noisy)
            {
                list.add(Move(square, one_step, FLAG_QUIET));
                if (relative_rank(side, square) == 1 && (empty & sq_bb(one_step + forward)))
                {
                    list.add(Move(square, one_step + forward, FLAG_DOUBLE));
                }
            }
        }
    }

    // --- Knights / King (leapers) ---
    Bitboard knights = pos.pieces(side, KNIGHT);
    while (knights)
    {
        int      square  = pop_lsb(knights);
        Bitboard targets = knight_attacks(square) & ~own;
        if (noisy)
        {
            targets &= enemy;
        }
        while (targets)
        {
            int to = pop_lsb(targets);
            list.add(Move(square, to, (enemy & sq_bb(to)) ? FLAG_CAPTURE : FLAG_QUIET));
        }
    }
    int king_square = pos.king_sq(side);
    {
        Bitboard targets = king_attacks(king_square) & ~own;
        if (noisy)
        {
            targets &= enemy;
        }
        while (targets)
        {
            int to = pop_lsb(targets);
            list.add(Move(king_square, to, (enemy & sq_bb(to)) ? FLAG_CAPTURE : FLAG_QUIET));
        }
    }

    // --- Sliders ---
    auto sliders = [&](Bitboard slider_pieces, auto attack_fn) {
        while (slider_pieces)
        {
            int      square  = pop_lsb(slider_pieces);
            Bitboard targets = attack_fn(square, occupancy) & ~own;
            if (noisy)
            {
                targets &= enemy;
            }
            while (targets)
            {
                int to = pop_lsb(targets);
                list.add(Move(square, to, (enemy & sq_bb(to)) ? FLAG_CAPTURE : FLAG_QUIET));
            }
        }
    };
    sliders(pos.pieces(side, BISHOP), bishop_attacks);
    sliders(pos.pieces(side, ROOK), rook_attacks);
    sliders(pos.pieces(side, QUEEN), queen_attacks);

    // --- Castling (fully legal: not in check, path empty and unattacked) ---
    if (!noisy && !pos.attacked_by(king_square, opponent))
    {
        int     rank           = side == WHITE ? 0 : 7;
        uint8_t kingside_flag  = side == WHITE ? CR_WK : CR_BK;
        uint8_t queenside_flag = side == WHITE ? CR_WQ : CR_BQ;
        int     e_square = make_square(4, rank), f_square = make_square(5, rank), g_square = make_square(6, rank);
        int     d_square = make_square(3, rank), c_square = make_square(2, rank), b_square = make_square(1, rank);
        if ((pos.castling & kingside_flag) && (empty & sq_bb(f_square)) && (empty & sq_bb(g_square)) &&
            !pos.attacked_by(f_square, opponent) && !pos.attacked_by(g_square, opponent))
        {
            list.add(Move(e_square, g_square, FLAG_KCASTLE));
        }
        if ((pos.castling & queenside_flag) && (empty & sq_bb(d_square)) && (empty & sq_bb(c_square)) &&
            (empty & sq_bb(b_square)) && !pos.attacked_by(d_square, opponent) && !pos.attacked_by(c_square, opponent))
        {
            list.add(Move(e_square, c_square, FLAG_QCASTLE));
        }
    }
}

void generate_legal(const Position &pos, MoveList &list, bool noisy_only)
{
    MoveList pseudo;
    generate_pseudo(pos, pseudo, noisy_only);
    for (Move move : pseudo)
    {
        if (pos.is_legal(move))
        {
            list.add(move);
        }
    }
}
