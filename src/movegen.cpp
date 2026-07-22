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
    const Bitboard own       = pos.byColor[side];
    const Bitboard enemy     = pos.byColor[opponent];
    const Bitboard empty     = ~occupancy;

    // --- Pawns (per-pawn for clarity; correctness before speed) ---
    Bitboard pawns   = pos.pieces(side, PAWN);
    int      forward = side == WHITE ? 8 : -8;
    while (pawns)
    {
        int  square    = pop_lsb(pawns);
        int  oneStep   = square + forward;
        bool promoRank = relative_rank(side, oneStep) == 7;

        // Captures + promotions on capture.
        Bitboard captures = pawn_attacks(side, square) & enemy;
        while (captures)
        {
            int to = pop_lsb(captures);
            if (promoRank)
            {
                add_promotions(list, square, to, true);
            }
            else
            {
                list.add(Move(square, to, FLAG_CAPTURE));
            }
        }
        // En passant.
        if (pos.epSq != NO_SQ && (pawn_attacks(side, square) & sq_bb(pos.epSq)))
        {
            list.add(Move(square, pos.epSq, FLAG_EP));
        }

        // Quiet pushes (skipped when generating noisy-only, except quiet promotions which are noisy).
        if (empty & sq_bb(oneStep))
        {
            if (promoRank)
            {
                add_promotions(list, square, oneStep, false);
            }
            else if (!noisy)
            {
                list.add(Move(square, oneStep, FLAG_QUIET));
                if (relative_rank(side, square) == 1 && (empty & sq_bb(oneStep + forward)))
                {
                    list.add(Move(square, oneStep + forward, FLAG_DOUBLE));
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
    int kingSquare = pos.king_sq(side);
    {
        Bitboard targets = king_attacks(kingSquare) & ~own;
        if (noisy)
        {
            targets &= enemy;
        }
        while (targets)
        {
            int to = pop_lsb(targets);
            list.add(Move(kingSquare, to, (enemy & sq_bb(to)) ? FLAG_CAPTURE : FLAG_QUIET));
        }
    }

    // --- Sliders ---
    auto sliders = [&](Bitboard sliderPieces, auto attackFn) {
        while (sliderPieces)
        {
            int      square  = pop_lsb(sliderPieces);
            Bitboard targets = attackFn(square, occupancy) & ~own;
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
    if (!noisy && !pos.attacked_by(kingSquare, opponent))
    {
        int     rank          = side == WHITE ? 0 : 7;
        uint8_t kingsideFlag  = side == WHITE ? CR_WK : CR_BK;
        uint8_t queensideFlag = side == WHITE ? CR_WQ : CR_BQ;
        int     eSquare = make_square(4, rank), fSquare = make_square(5, rank), gSquare = make_square(6, rank);
        int     dSquare = make_square(3, rank), cSquare = make_square(2, rank), bSquare = make_square(1, rank);
        if ((pos.castling & kingsideFlag) && (empty & sq_bb(fSquare)) && (empty & sq_bb(gSquare)) &&
            !pos.attacked_by(fSquare, opponent) && !pos.attacked_by(gSquare, opponent))
        {
            list.add(Move(eSquare, gSquare, FLAG_KCASTLE));
        }
        if ((pos.castling & queensideFlag) && (empty & sq_bb(dSquare)) && (empty & sq_bb(cSquare)) &&
            (empty & sq_bb(bSquare)) && !pos.attacked_by(dSquare, opponent) && !pos.attacked_by(cSquare, opponent))
        {
            list.add(Move(eSquare, cSquare, FLAG_QCASTLE));
        }
    }
}

void generate_legal(const Position &pos, MoveList &list, bool noisyOnly)
{
    MoveList pseudo;
    generate_pseudo(pos, pseudo, noisyOnly);
    for (Move move : pseudo)
    {
        if (pos.is_legal(move))
        {
            list.add(move);
        }
    }
}
