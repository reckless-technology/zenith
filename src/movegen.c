// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief Move generation: the pseudo-legal generator and the legal filter built on top of it.
 */
#include "movegen.h"
#include "bitboard.h"

/** @brief Emit all four promotions for a pawn reaching the last rank (queen first for move ordering). */
static void add_promotions(MoveList *list, const int from, const int to, const bool is_capture)
{
    const unsigned base = is_capture ? FLAG_PROMOTION_KNIGHT_CAPTURE : FLAG_PROMOTION_KNIGHT;
    // Queen first (best for move ordering), then knight/rook/bishop.
    movelist_add(list, move_make(from, to, base + (QUEEN - KNIGHT)));
    movelist_add(list, move_make(from, to, base + (KNIGHT - KNIGHT)));
    movelist_add(list, move_make(from, to, base + (ROOK - KNIGHT)));
    movelist_add(list, move_make(from, to, base + (BISHOP - KNIGHT)));
}

/** @brief Emit every slider move for one piece type given its attack function (is_noisy_only = captures only). */
static void slider_moves(MoveList *list, Bitboard slider_pieces, Bitboard (*attack_fn)(int, Bitboard),
                         const Bitboard own, const Bitboard enemy, const Bitboard occupancy, const bool is_noisy_only)
{
    while (slider_pieces)
    {
        const int square  = pop_lsb(&slider_pieces);
        Bitboard  targets = attack_fn(square, occupancy) & ~own;
        if (is_noisy_only)
        {
            targets &= enemy;
        }
        while (targets)
        {
            const int to = pop_lsb(&targets);
            movelist_add(list, move_make(square, to, (enemy & sq_bb(to)) ? FLAG_CAPTURE : FLAG_QUIET));
        }
    }
}

/**
 * @brief Generate pseudo-legal moves into @p list (which is reset here).
 *
 * Castling is emitted fully legal (king not in/through check); every other move is legal iff it does not
 * leave the mover's own king in check — the search tests that with a single make_move (see negamax/qsearch),
 * avoiding the separate copy-make that generate_legal does per move.
 */
void generate_pseudo(const Position *pos, MoveList *list, bool is_noisy_only)
{
    list->count = 0;

    const Color    side = pos->color_to_move, opponent = enemy_of(side);
    const Bitboard occupancy = position_occupied(pos);
    const Bitboard own       = pos->colors[side];
    const Bitboard enemy     = pos->colors[opponent];
    const Bitboard empty     = ~occupancy;

    // --- Pawns (per-pawn for clarity; correctness before speed) ---
    Bitboard  pawns   = position_pieces(pos, side, PAWN);
    const int forward = side == WHITE ? 8 : -8;
    while (pawns)
    {
        const int  square        = pop_lsb(&pawns);
        const int  one_step      = square + forward;
        const bool is_promo_rank = relative_rank(side, one_step) == 7;

        // Captures + promotions on capture.
        Bitboard captures = pawn_attacks(side, square) & enemy;
        while (captures)
        {
            const int to = pop_lsb(&captures);
            if (is_promo_rank)
            {
                add_promotions(list, square, to, true);
            }
            else
            {
                movelist_add(list, move_make(square, to, FLAG_CAPTURE));
            }
        }
        // En passant.
        if (pos->ep_square != NO_SQUARE && (pawn_attacks(side, square) & sq_bb(pos->ep_square)))
        {
            movelist_add(list, move_make(square, pos->ep_square, FLAG_EP));
        }

        // Quiet pushes (skipped when generating noisy-only, except quiet promotions which are noisy).
        if (empty & sq_bb(one_step))
        {
            if (is_promo_rank)
            {
                add_promotions(list, square, one_step, false);
            }
            else if (!is_noisy_only)
            {
                movelist_add(list, move_make(square, one_step, FLAG_QUIET));
                if (relative_rank(side, square) == 1 && (empty & sq_bb(one_step + forward)))
                {
                    movelist_add(list, move_make(square, one_step + forward, FLAG_PAWN_DOUBLE_PUSH));
                }
            }
        }
    }

    // --- Knights / King (leapers) ---
    Bitboard knights = position_pieces(pos, side, KNIGHT);
    while (knights)
    {
        const int square  = pop_lsb(&knights);
        Bitboard  targets = knight_attacks(square) & ~own;
        if (is_noisy_only)
        {
            targets &= enemy;
        }
        while (targets)
        {
            const int to = pop_lsb(&targets);
            movelist_add(list, move_make(square, to, (enemy & sq_bb(to)) ? FLAG_CAPTURE : FLAG_QUIET));
        }
    }
    const int king_square = position_king_sq(pos, side);
    {
        Bitboard targets = king_attacks(king_square) & ~own;
        if (is_noisy_only)
        {
            targets &= enemy;
        }
        while (targets)
        {
            const int to = pop_lsb(&targets);
            movelist_add(list, move_make(king_square, to, (enemy & sq_bb(to)) ? FLAG_CAPTURE : FLAG_QUIET));
        }
    }

    // --- Sliders ---
    slider_moves(list, position_pieces(pos, side, BISHOP), bishop_attacks, own, enemy, occupancy, is_noisy_only);
    slider_moves(list, position_pieces(pos, side, ROOK), rook_attacks, own, enemy, occupancy, is_noisy_only);
    slider_moves(list, position_pieces(pos, side, QUEEN), queen_attacks, own, enemy, occupancy, is_noisy_only);

    // --- Castling (fully legal: not in check, path empty and unattacked) ---
    if (!is_noisy_only && !position_is_attacked_by(pos, king_square, opponent))
    {
        const int     rank           = side == WHITE ? 0 : 7;
        const uint8_t kingside_flag  = side == WHITE ? MAY_WHITE_CASTLE_KINGSIDE : MAY_BLACK_CASTLE_KINGSIDE;
        const uint8_t queenside_flag = side == WHITE ? MAY_WHITE_CASTLE_QUEENSIDE : MAY_BLACK_CASTLE_QUEENSIDE;
        const int     e_square = make_square(4, rank), f_square = make_square(5, rank), g_square = make_square(6, rank);
        const int     d_square = make_square(3, rank), c_square = make_square(2, rank), b_square = make_square(1, rank);
        if ((pos->castling_rights & kingside_flag) && (empty & sq_bb(f_square)) && (empty & sq_bb(g_square)) &&
            !position_is_attacked_by(pos, f_square, opponent) && !position_is_attacked_by(pos, g_square, opponent))
        {
            movelist_add(list, move_make(e_square, g_square, FLAG_CASTLE_KINGSIDE));
        }
        if ((pos->castling_rights & queenside_flag) && (empty & sq_bb(d_square)) && (empty & sq_bb(c_square)) &&
            (empty & sq_bb(b_square)) && !position_is_attacked_by(pos, d_square, opponent) &&
            !position_is_attacked_by(pos, c_square, opponent))
        {
            movelist_add(list, move_make(e_square, c_square, FLAG_CASTLE_QUEENSIDE));
        }
    }
}

/** @brief Generate fully legal moves: pseudo-legal generation followed by a copy-make legality filter. */
void generate_legal(const Position *pos, MoveList *list, bool is_noisy_only)
{
    list->count = 0;

    MoveList pseudo;
    generate_pseudo(pos, &pseudo, is_noisy_only);
    for (int index = 0; index < pseudo.count; index++)
    {
        const Move move = pseudo.moves[index];
        if (position_is_legal(pos, move))
        {
            movelist_add(list, move);
        }
    }
}
