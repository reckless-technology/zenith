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

    // --- Pawns (setwise: one bitboard shift per move kind generates every pawn's move in parallel) ---
    const Bitboard pawns      = position_pieces(pos, side, PAWN);
    const Bitboard promo_rank = side == WHITE ? RANK_8 : RANK_1;
    const int      forward    = side == WHITE ? 8 : -8;

    // Captures: shift the whole pawn set toward each capture diagonal; the shift helpers mask file wraps.
    const Bitboard east_targets = (side == WHITE ? shift_ne(pawns) : shift_se(pawns)) & enemy;
    const Bitboard west_targets = (side == WHITE ? shift_nw(pawns) : shift_sw(pawns)) & enemy;
    const int      east_delta = side == WHITE ? 9 : -7, west_delta = side == WHITE ? 7 : -9;

    Bitboard east_promos = east_targets & promo_rank;
    while (east_promos)
    {
        const int to = pop_lsb(&east_promos);
        add_promotions(list, to - east_delta, to, true);
    }
    Bitboard east_captures = east_targets & ~promo_rank;
    while (east_captures)
    {
        const int to = pop_lsb(&east_captures);
        movelist_add(list, move_make(to - east_delta, to, FLAG_CAPTURE));
    }
    Bitboard west_promos = west_targets & promo_rank;
    while (west_promos)
    {
        const int to = pop_lsb(&west_promos);
        add_promotions(list, to - west_delta, to, true);
    }
    Bitboard west_captures = west_targets & ~promo_rank;
    while (west_captures)
    {
        const int to = pop_lsb(&west_captures);
        movelist_add(list, move_make(to - west_delta, to, FLAG_CAPTURE));
    }

    // En passant: the pawns attacking the ep square are exactly the squares an enemy pawn there would attack.
    if (pos->ep_square != NO_SQUARE)
    {
        Bitboard ep_attackers = pawn_attacks(opponent, pos->ep_square) & pawns;
        while (ep_attackers)
        {
            const int from = pop_lsb(&ep_attackers);
            movelist_add(list, move_make(from, pos->ep_square, FLAG_EP));
        }
    }

    // Pushes: single, promotion, and double, each derived from one shifted set.
    const Bitboard single_targets = (side == WHITE ? shift_north(pawns) : shift_south(pawns)) & empty;

    // Promotion pushes are noisy — emitted even in noisy-only generation, like promotion captures.
    Bitboard push_promos = single_targets & promo_rank;
    while (push_promos)
    {
        const int to = pop_lsb(&push_promos);
        add_promotions(list, to - forward, to, false);
    }
    if (!is_noisy_only)
    {
        Bitboard single_pushes = single_targets & ~promo_rank;
        while (single_pushes)
        {
            const int to = pop_lsb(&single_pushes);
            movelist_add(list, move_make(to - forward, to, FLAG_QUIET));
        }
        // Double pushes: shift the single-push set once more and keep only the double-push landing rank
        // (rank 4 / rank 5) — that restricts them to pawns that started on the home rank with a clear path.
        const Bitboard double_rank = side == WHITE ? RANK_4 : RANK_5;
        Bitboard       double_pushes =
            (side == WHITE ? shift_north(single_targets) : shift_south(single_targets)) & empty & double_rank;
        while (double_pushes)
        {
            const int to = pop_lsb(&double_pushes);
            movelist_add(list, move_make(to - 2 * forward, to, FLAG_PAWN_DOUBLE_PUSH));
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
