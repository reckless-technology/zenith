// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief Move generation: one masked setwise skeleton serving both the pseudo-legal and the legal generator.
 *
 * generate_moves emits moves whose targets are pre-intersected with a check-evasion mask and per-piece pin
 * rays. generate_pseudo instantiates it with permissive masks (every mask a no-op — emissions identical to a
 * plain pseudo-legal generator); generate_legal instantiates it with the real masks, so legality is baked into
 * generation and no per-move filter pass is needed (king moves get a per-destination safety test; en passant,
 * the one case masks cannot express, gets the full copy-free test). Both instantiations are inlined wrappers,
 * so the compiler constant-folds the mode flag and the pseudo path costs nothing extra.
 *
 * Emission is a plain unchecked cursor (`*out++ = move`): MAX_MOVES is provably sufficient for any position
 * position_set_fen accepts (see movegen.h), which the debug build re-checks with an assert.
 */
#include "movegen.h"
#include "bitboard.h"
#include <assert.h>

/** @brief Emit all four promotions for a pawn reaching the last rank (queen first for move ordering). */
static inline Move *add_promotions(Move *out, const int from, const int to, const bool is_capture)
{
    const unsigned base = is_capture ? FLAG_PROMOTION_KNIGHT_CAPTURE : FLAG_PROMOTION_KNIGHT;
    // Queen first (best for move ordering), then knight/rook/bishop.
    *out++ = move_make(from, to, base + (QUEEN - KNIGHT));
    *out++ = move_make(from, to, base + (KNIGHT - KNIGHT));
    *out++ = move_make(from, to, base + (ROOK - KNIGHT));
    *out++ = move_make(from, to, base + (BISHOP - KNIGHT));
    return out;
}

/**
 * @brief Emit every slider move for one piece type given its attack function (is_noisy_only = captures only).
 *
 * Targets are intersected with @p check_mask; a pinned slider is further restricted to its pin ray (the line
 * through it and the king). With permissive masks (check_mask = ~0, pinned = 0) this is the plain pseudo-legal
 * emission.
 */
static Move *slider_moves(Move *out, Bitboard slider_pieces, Bitboard (*attack_fn)(int, Bitboard), const Bitboard own,
                          const Bitboard enemy, const Bitboard occupancy, const bool is_noisy_only,
                          const Bitboard check_mask, const Bitboard pinned, const int king_square)
{
    while (slider_pieces)
    {
        const int square  = pop_lsb(&slider_pieces);
        Bitboard  targets = attack_fn(square, occupancy) & ~own & check_mask;
        if (pinned & sq_bb(square))
        {
            targets &= line_bb(king_square, square);
        }
        if (is_noisy_only)
        {
            targets &= enemy;
        }
        while (targets)
        {
            const int to = pop_lsb(&targets);
            *out++       = move_make(square, to, (enemy & sq_bb(to)) ? FLAG_CAPTURE : FLAG_QUIET);
        }
    }
    return out;
}

/**
 * @brief The masked setwise generator both public generators instantiate (always inlined into its two
 * wrappers, so @p is_legal_mode and the permissive masks constant-fold away in the pseudo instantiation).
 *
 * @param pos the position.
 * @param moves output buffer (MAX_MOVES), MOVE_NONE-terminated.
 * @param is_noisy_only captures + promotions only.
 * @param is_legal_mode true = emit fully legal moves (per-destination king safety, full en-passant test);
 *        false = pseudo-legal (callers filter with position_is_move_legal).
 * @param check_mask target filter for non-king moves: ~0 when not in check, checker|blocking-ray in single
 *        check, 0 in double check (only king moves can resolve).
 * @param pinned our pieces pinned to our king (position_pinned_to_king); pinned targets are restricted to
 *        the pin ray. Pass 0 in pseudo mode.
 * @return the number of moves emitted.
 */
static inline int generate_moves(const Position *pos, Move *moves, const bool is_noisy_only, const bool is_legal_mode,
                                 const Bitboard check_mask, const Bitboard pinned)
{
    Move *out = moves;

    const Color    side = pos->color_to_move, opponent = enemy_of(side);
    const Bitboard occupancy   = position_occupied(pos);
    const Bitboard own         = pos->colors[side];
    const Bitboard enemy       = pos->colors[opponent];
    const Bitboard empty       = ~occupancy;
    const int      king_square = pos->king_location[side];

    // --- Pawns (setwise: one bitboard shift per move kind generates every pawn's move in parallel) ---
    // Pin rays are per-pawn, so the setwise shifts cover only unpinned pawns; the (rare) pinned pawns are
    // generated individually below. In pseudo mode pinned = 0 and both adjustments vanish.
    const Bitboard pawns      = position_pieces(pos, side, PAWN) & ~pinned;
    const Bitboard promo_rank = side == WHITE ? RANK_8 : RANK_1;
    const int      forward    = side == WHITE ? 8 : -8;

    // Captures: shift the whole pawn set toward each capture diagonal; the shift helpers mask file wraps.
    const Bitboard east_targets =
        (side == WHITE ? shift_northeast(pawns) : shift_southeast(pawns)) & enemy & check_mask;
    const Bitboard west_targets =
        (side == WHITE ? shift_northwest(pawns) : shift_southwest(pawns)) & enemy & check_mask;
    const int east_delta = side == WHITE ? 9 : -7, west_delta = side == WHITE ? 7 : -9;

    Bitboard east_promos = east_targets & promo_rank;
    while (east_promos)
    {
        const int to = pop_lsb(&east_promos);
        out          = add_promotions(out, to - east_delta, to, true);
    }
    Bitboard east_captures = east_targets & ~promo_rank;
    while (east_captures)
    {
        const int to = pop_lsb(&east_captures);
        *out++       = move_make(to - east_delta, to, FLAG_CAPTURE);
    }
    Bitboard west_promos = west_targets & promo_rank;
    while (west_promos)
    {
        const int to = pop_lsb(&west_promos);
        out          = add_promotions(out, to - west_delta, to, true);
    }
    Bitboard west_captures = west_targets & ~promo_rank;
    while (west_captures)
    {
        const int to = pop_lsb(&west_captures);
        *out++       = move_make(to - west_delta, to, FLAG_CAPTURE);
    }

    // En passant: the pawns attacking the ep square are exactly the squares an enemy pawn there would attack.
    // Check evasion and pins cannot be expressed as a target mask here (the captured pawn leaves a different
    // square than the ep target), so legal mode runs the full copy-free ep test instead — the mover may even
    // be a "pinned" pawn whose capture stays legal, which is why ep candidates come from ALL our pawns.
    if (pos->ep_square != NO_SQUARE)
    {
        Bitboard ep_attackers = pawn_attacks(opponent, pos->ep_square) & position_pieces(pos, side, PAWN);
        while (ep_attackers)
        {
            const int  from    = pop_lsb(&ep_attackers);
            const Move ep_move = move_make(from, pos->ep_square, FLAG_EP);
            if (is_legal_mode && !position_is_move_legal(pos, ep_move, pinned))
            {
                continue;
            }
            *out++ = ep_move;
        }
    }

    // Pushes: single, promotion, and double, each derived from one shifted set. The double-push set must be
    // derived from the UNMASKED single-push squares (a double push may block a check even though the skipped
    // square is not on the evasion mask).
    const Bitboard single_targets = (side == WHITE ? shift_north(pawns) : shift_south(pawns)) & empty;

    // Promotion pushes are noisy — emitted even in noisy-only generation, like promotion captures.
    Bitboard push_promos = single_targets & promo_rank & check_mask;
    while (push_promos)
    {
        const int to = pop_lsb(&push_promos);
        out          = add_promotions(out, to - forward, to, false);
    }
    if (!is_noisy_only)
    {
        Bitboard single_pushes = single_targets & ~promo_rank & check_mask;
        while (single_pushes)
        {
            const int to = pop_lsb(&single_pushes);
            *out++       = move_make(to - forward, to, FLAG_QUIET);
        }
        // Double pushes: shift the single-push set once more and keep only the double-push landing rank
        // (rank 4 / rank 5) — that restricts them to pawns that started on the home rank with a clear path.
        const Bitboard double_rank = side == WHITE ? RANK_4 : RANK_5;
        Bitboard double_pushes = (side == WHITE ? shift_north(single_targets) : shift_south(single_targets)) & empty &
                                 double_rank & check_mask;
        while (double_pushes)
        {
            const int to = pop_lsb(&double_pushes);
            *out++       = move_make(to - 2 * forward, to, FLAG_PAWN_DOUBLE_PUSH);
        }
    }

    // Pinned pawns, one at a time: every target is confined to the pin ray (which also permits capturing the
    // pinner, including with promotion). An empty set in pseudo mode — the loop never runs.
    Bitboard pinned_pawns = position_pieces(pos, side, PAWN) & pinned;
    while (pinned_pawns)
    {
        const int      from    = pop_lsb(&pinned_pawns);
        const Bitboard allowed = line_bb(king_square, from) & check_mask;

        Bitboard captures = pawn_attacks(side, from) & enemy & allowed;
        while (captures)
        {
            const int to = pop_lsb(&captures);
            if (promo_rank & sq_bb(to))
            {
                out = add_promotions(out, from, to, true);
            }
            else
            {
                *out++ = move_make(from, to, FLAG_CAPTURE);
            }
        }

        const Bitboard single = (side == WHITE ? shift_north(sq_bb(from)) : shift_south(sq_bb(from))) & empty;
        if (single & promo_rank & allowed)
        {
            out = add_promotions(out, from, from + forward, false);
        }
        if (!is_noisy_only)
        {
            if (single & ~promo_rank & allowed)
            {
                *out++ = move_make(from, from + forward, FLAG_QUIET);
            }
            const Bitboard double_rank = side == WHITE ? RANK_4 : RANK_5;
            if ((side == WHITE ? shift_north(single) : shift_south(single)) & empty & double_rank & allowed)
            {
                *out++ = move_make(from, from + 2 * forward, FLAG_PAWN_DOUBLE_PUSH);
            }
        }
    }

    // --- Knights (a pinned knight never has a legal move: knight moves always leave the pin ray) ---
    Bitboard knights = position_pieces(pos, side, KNIGHT) & ~pinned;
    while (knights)
    {
        const int square  = pop_lsb(&knights);
        Bitboard  targets = knight_attacks(square) & ~own & check_mask;
        if (is_noisy_only)
        {
            targets &= enemy;
        }
        while (targets)
        {
            const int to = pop_lsb(&targets);
            *out++       = move_make(square, to, (enemy & sq_bb(to)) ? FLAG_CAPTURE : FLAG_QUIET);
        }
    }

    // --- King (never masked by check_mask — escaping is the point; legal mode tests each destination for
    // safety on the occupancy WITHOUT the king, so a slider's ray extends through the vacated square) ---
    {
        Bitboard targets = king_attacks(king_square) & ~own;
        if (is_noisy_only)
        {
            targets &= enemy;
        }
        while (targets)
        {
            const int to = pop_lsb(&targets);
            if (is_legal_mode && position_attackers_to(pos, to, opponent, occupancy ^ sq_bb(king_square)))
            {
                continue;
            }
            *out++ = move_make(king_square, to, (enemy & sq_bb(to)) ? FLAG_CAPTURE : FLAG_QUIET);
        }
    }

    // --- Sliders ---
    out = slider_moves(out, position_pieces(pos, side, BISHOP), bishop_attacks, own, enemy, occupancy, is_noisy_only,
                       check_mask, pinned, king_square);
    out = slider_moves(out, position_pieces(pos, side, ROOK), rook_attacks, own, enemy, occupancy, is_noisy_only,
                       check_mask, pinned, king_square);
    out = slider_moves(out, position_pieces(pos, side, QUEEN), queen_attacks, own, enemy, occupancy, is_noisy_only,
                       check_mask, pinned, king_square);

    // --- Castling (fully legal in both modes: not in check, path empty and unattacked). The cached checkers
    // set is exactly "attackers to our king", so the not-in-check gate is a single load. ---
    if (!is_noisy_only && pos->checkers == 0)
    {
        const int     rank           = side == WHITE ? 0 : 7;
        const uint8_t kingside_flag  = side == WHITE ? MAY_WHITE_CASTLE_KINGSIDE : MAY_BLACK_CASTLE_KINGSIDE;
        const uint8_t queenside_flag = side == WHITE ? MAY_WHITE_CASTLE_QUEENSIDE : MAY_BLACK_CASTLE_QUEENSIDE;
        const int     e_square = make_square(4, rank), f_square = make_square(5, rank), g_square = make_square(6, rank);
        const int     d_square = make_square(3, rank), c_square = make_square(2, rank), b_square = make_square(1, rank);
        if ((pos->castling_rights & kingside_flag) && (empty & sq_bb(f_square)) && (empty & sq_bb(g_square)) &&
            !position_is_attacked_by(pos, f_square, opponent) && !position_is_attacked_by(pos, g_square, opponent))
        {
            *out++ = move_make(e_square, g_square, FLAG_CASTLE);
        }
        if ((pos->castling_rights & queenside_flag) && (empty & sq_bb(d_square)) && (empty & sq_bb(c_square)) &&
            (empty & sq_bb(b_square)) && !position_is_attacked_by(pos, d_square, opponent) &&
            !position_is_attacked_by(pos, c_square, opponent))
        {
            *out++ = move_make(e_square, c_square, FLAG_CASTLE);
        }
    }

    assert(out - moves < MAX_MOVES); // the movegen.h bound: <= 415 moves + the terminator slot
    *out = MOVE_NONE;                // sentinel terminator
    return (int)(out - moves);
}

/**
 * @brief Generate pseudo-legal moves into @p moves (a MAX_MOVES buffer), terminated with MOVE_NONE.
 *
 * Castling is emitted fully legal (king not in/through check); every other move is legal iff it does not
 * leave the mover's own king in check — the search filters that with the copy-free position_is_move_legal before
 * pruning/make. The permissive masks make this instantiation emit exactly the plain pseudo-legal move set.
 */
int generate_pseudo(const Position *pos, Move *moves, bool is_noisy_only)
{
    return generate_moves(pos, moves, is_noisy_only, false, ~0ULL, 0);
}

/**
 * @brief Generate fully legal moves: the masked generator with the real check-evasion and pin masks, so no
 * per-move filter pass runs (this is what makes perft/datagen/UCI parsing fast).
 */
int generate_legal(const Position *pos, Move *moves, bool is_noisy_only)
{
    const Bitboard checkers    = pos->checkers;
    const Bitboard pinned      = position_pinned_to_king(pos);
    const int      king_square = pos->king_location[pos->color_to_move];

    // Not in check: any target. Single check: capture the checker or block its ray. Double check: no
    // non-king move helps (king moves are never masked; en passant runs its own full test).
    Bitboard check_mask = ~0ULL;
    if (checkers)
    {
        check_mask = has_more_than_one(checkers) ? 0 : (checkers | between_bb(king_square, lsb(checkers)));
    }
    return generate_moves(pos, moves, is_noisy_only, true, check_mask, pinned);
}
