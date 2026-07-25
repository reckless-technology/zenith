// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief The search: iterative deepening, PVS negamax, quiescence, SEE, and all pruning/ordering heuristics.
 */
#include "search.h"
#include "bitboard.h"
#include "eval.h"
#include "nnue.h"
#include "platform.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

atomic_bool  g_stop   = false;
SearchParams g_params = {
    .rfp_margin         = 62,
    .nmp_divisor        = 202,
    .lmp_base           = 4,
    .futility_base      = 102,
    .futility_margin    = 96,
    .see_capture_margin = 102,
    .lmr_base_x100      = 86,
    .lmr_divisor_x100   = 229,
    .singular_margin    = 3,
    .aspiration_delta   = 21,
    .history_max        = 418,
};

static int       Reductions[MAX_PLY][64];
static const int SeeValue[NUM_PIECES] = {0, 100, 320, 330, 500, 900, 10000}; // indexed by Piece

/**
 * @brief Eval correction-history scale constants.
 *
 * A running average of (search score - static eval) keyed by pawn structure, used to nudge the static eval
 * toward what search has historically found. Entries are cp * CORRHIST_GRAIN.
 */
enum
{
    CORRHIST_SIZE  = 16384,              ///< power of two -> index by (pawn_key & (SIZE-1))
    CORRHIST_GRAIN = 256,                ///< fixed-point scale of a stored correction
    CORRHIST_MAX   = 64 * CORRHIST_GRAIN ///< clamp the correction to +/-64 cp
};

static int min_int(const int a, const int b)
{
    return a < b ? a : b;
}

static int max_int(const int a, const int b)
{
    return a > b ? a : b;
}

static int clamp_int(const int value, const int low, const int high)
{
    return value < low ? low : (value > high ? high : value);
}

static int64_t max_i64(const int64_t a, const int64_t b)
{
    return a > b ? a : b;
}

static int64_t min_i64(const int64_t a, const int64_t b)
{
    return a < b ? a : b;
}

static int draw_value(void)
{
    return 0;
}

/**
 * @brief All pieces of either color attacking @p square, given @p occupancy.
 *
 * Equivalent to position_attackers_to(WHITE) | position_attackers_to(BLACK), but does each slider magic
 * lookup once instead of once per color — the exchange loop below recomputes this every ply.
 */
static inline Bitboard attackers_to_both(const Position *pos, const int square, const Bitboard occupancy)
{
    return (pawn_attacks(BLACK, square) & position_pieces(pos, WHITE, PAWN)) |
           (pawn_attacks(WHITE, square) & position_pieces(pos, BLACK, PAWN)) |
           (knight_attacks(square) & pos->pieces[KNIGHT]) | (king_attacks(square) & pos->pieces[KING]) |
           (bishop_attacks(square, occupancy) & (pos->pieces[BISHOP] | pos->pieces[QUEEN])) |
           (rook_attacks(square, occupancy) & (pos->pieces[ROOK] | pos->pieces[QUEEN]));
}

/** @brief SEE of a capture: net material after the optimal capture sequence on the target square. */
int static_exchange_eval(const Position *pos, const Move move)
{
    const int to = move_to(move), from = move_from(move);
    const int captured = move_is_ep(move) ? SeeValue[PAWN] : SeeValue[pos->board[to]];
    int       gain[32];
    int       swap_index = 0;
    gain[0]              = captured;

    Bitboard occupied = position_occupied(pos);
    if (move_is_ep(move))
    {
        occupied ^= sq_bb(to + (pos->color_to_move == WHITE ? -8 : 8));
    }
    occupied ^= sq_bb(from);

    Piece attacker = (Piece)pos->board[from];
    Color side     = enemy_of(pos->color_to_move);
    // Attackers hitting `to` given the current occupancy; recomputed each ply so x-rays reveal.
    Bitboard attackers = attackers_to_both(pos, to, occupied) & occupied;

    while (true)
    {
        swap_index++;
        if (swap_index >= 32)
        {
            break; // gain[32] guard: real positions never approach 32 recaptures, but a crafted illegal
                   // FEN with >32 attackers on one square could otherwise overflow the array.
        }
        gain[swap_index]              = SeeValue[attacker] - gain[swap_index - 1];
        const Bitboard side_attackers = attackers & pos->colors[side];
        if (!side_attackers)
        {
            break;
        }
        // Least valuable attacker of `side`.
        Piece    least_valuable_attacker = KING;
        Bitboard attacker_bit            = 0;
        for (int piece = PAWN; piece <= KING; piece++)
        {
            const Bitboard candidates = side_attackers & position_pieces(pos, side, (Piece)piece);
            if (candidates)
            {
                least_valuable_attacker = (Piece)piece;
                attacker_bit            = candidates & (~candidates + 1);
                break;
            }
        }
        attacker = least_valuable_attacker;
        occupied ^= attacker_bit;
        attackers = attackers_to_both(pos, to, occupied) & occupied;
        side      = enemy_of(side);
        if (attacker == KING && (attackers & pos->colors[side]))
        {
            // Cannot recapture with the king into a still-defended square; stop before it.
            break;
        }
    }
    while (--swap_index > 0)
    {
        gain[swap_index - 1] = -max_int(-gain[swap_index - 1], gain[swap_index]);
    }
    return gain[0];
}

void init_search(void)
{
    const double lmr_base    = g_params.lmr_base_x100 / 100.0;
    const double lmr_divisor = g_params.lmr_divisor_x100 / 100.0;
    for (int depth = 1; depth < MAX_PLY; depth++)
    {
        for (int move_number = 1; move_number < 64; move_number++)
        {
            Reductions[depth][move_number] = (int)(lmr_base + log(depth) * log(move_number) / lmr_divisor);
        }
    }
}

bool set_search_param(const char *name, const int value)
{
    if (strcmp(name, "RfpMargin") == 0)
    {
        g_params.rfp_margin = value;
    }
    else if (strcmp(name, "NmpDivisor") == 0)
    {
        g_params.nmp_divisor = value;
    }
    else if (strcmp(name, "LmpBase") == 0)
    {
        g_params.lmp_base = value;
    }
    else if (strcmp(name, "FutilityBase") == 0)
    {
        g_params.futility_base = value;
    }
    else if (strcmp(name, "FutilityMargin") == 0)
    {
        g_params.futility_margin = value;
    }
    else if (strcmp(name, "SeeCaptureMargin") == 0)
    {
        g_params.see_capture_margin = value;
    }
    else if (strcmp(name, "SingularMargin") == 0)
    {
        g_params.singular_margin = value;
    }
    else if (strcmp(name, "AspirationDelta") == 0)
    {
        g_params.aspiration_delta = value;
    }
    else if (strcmp(name, "HistoryMax") == 0)
    {
        g_params.history_max = value;
    }
    else if (strcmp(name, "LmrBase") == 0)
    {
        g_params.lmr_base_x100 = value;
        init_search(); // LMR table depends on this
    }
    else if (strcmp(name, "LmrDivisor") == 0)
    {
        g_params.lmr_divisor_x100 = value;
        init_search();
    }
    else
    {
        return false;
    }
    return true;
}

static int64_t elapsed(const Searcher *searcher)
{
    return platform_now_ms() - searcher->start_ms;
}

/** @brief Whether the search must stop now (g_stop set, or the main thread hit its node/time budget). */
static bool is_time_up(const Searcher *searcher)
{
    if (atomic_load_explicit(&g_stop, memory_order_relaxed))
    {
        return true;
    }
    // Only the main thread owns the time/node budget; when it runs out it sets g_stop so helpers stop too.
    if (searcher->is_main && ((searcher->node_limit && searcher->nodes >= (uint64_t)searcher->node_limit) ||
                              (searcher->is_time_limited && elapsed(searcher) >= searcher->hard_ms)))
    {
        atomic_store_explicit(&g_stop, true, memory_order_relaxed);
        return true;
    }
    return false;
}

/** @brief Compute this search's soft/hard time budgets and node cap from @p lim and the root side's clock. */
static void set_time(Searcher *searcher, const Position *root, const SearchLimits *lim)
{
    searcher->is_time_limited = false;
    searcher->soft_ms = searcher->hard_ms = 0;
    searcher->node_limit                  = lim->nodes;
    if (lim->movetime > 0)
    {
        searcher->is_time_limited = true;
        searcher->hard_ms = searcher->soft_ms = max_i64(1, lim->movetime - searcher->move_overhead);
    }
    else if (lim->time[root->color_to_move] > 0)
    {
        searcher->is_time_limited = true;
        const int64_t remaining = lim->time[root->color_to_move], increment = lim->inc[root->color_to_move];
        const int     moves_to_go = lim->movestogo > 0 ? lim->movestogo : 30;
        const int64_t budget      = remaining / moves_to_go + increment * 3 / 4;
        searcher->soft_ms         = max_i64(1, budget - searcher->move_overhead);
        searcher->hard_ms         = max_i64(1, min_i64(remaining - searcher->move_overhead, searcher->soft_ms * 4));
    }
    else if (lim->has_time_control)
    {
        // A clock/movetime token was given but the side-to-move time is non-positive (flag fall, or a GUI
        // that sent a zero/negative time). Move immediately on a minimal budget instead of searching
        // unbounded — the distinction from a bare `go` needs the explicit flag (wtime 0 reads as time 0).
        searcher->is_time_limited = true;
        searcher->soft_ms = searcher->hard_ms = 1;
    }
}

/** @brief Whether @p pos is a draw by 50-move rule, insufficient material, or repetition within the history. */
static bool is_draw(const Searcher *searcher, const Position *pos)
{
    if (pos->halfmove >= 100)
    {
        return true;
    }
    // Insufficient material (K vs K, K+minor vs K/K+minor).
    if (!(pos->pieces[PAWN] | pos->pieces[ROOK] | pos->pieces[QUEEN]))
    {
        const int white_minors = popcount(pos->colors[WHITE] & (pos->pieces[KNIGHT] | pos->pieces[BISHOP]));
        const int black_minors = popcount(pos->colors[BLACK] & (pos->pieces[KNIGHT] | pos->pieces[BISHOP]));
        if (white_minors <= 1 && black_minors <= 1)
        {
            return true;
        }
    }
    // Repetition: scan the path back to the last irreversible move (step 2 keeps the side to move).
    const int end     = searcher->hist_count;
    const int stop_at = max_int(0, end - pos->halfmove);
    for (int i = end - 2; i >= stop_at; i -= 2)
    {
        if (searcher->hist_keys[i] == pos->key)
        {
            return true;
        }
    }
    return false;
}

static void update_pv(Searcher *searcher, const int ply, const Move move)
{
    searcher->pv_table[ply][0] = move;
    memcpy(&searcher->pv_table[ply][1], &searcher->pv_table[ply + 1][0], searcher->pv_len[ply + 1] * sizeof(Move));
    searcher->pv_len[ply] = searcher->pv_len[ply + 1] + 1;
}

// History gravity: saturating blend toward the new bonus (larger entries move less).
static inline void apply_gravity(int *entry, const int change)
{
    *entry += change - *entry * abs(change) / 16384;
}

/** @brief Quiescence search: extend the leaf with captures/promotions (SEE-pruned) until the position is quiet. */
static int qsearch(Searcher *searcher, const Position *pos, int alpha, const int beta, const int ply)
{
    if (is_time_up(searcher))
    {
        return 0; // is_time_up() already set g_stop for the main thread
    }
    searcher->nodes++;
    if (ply > searcher->seldepth)
    {
        searcher->seldepth = ply;
    }
    if (ply >= MAX_PLY - 1)
    {
        return evaluate(pos);
    }

    const Bitboard checkers    = position_attackers_to(pos, position_king_sq(pos, pos->color_to_move),
                                                       enemy_of(pos->color_to_move), position_occupied(pos));
    const bool     is_in_check = checkers != 0;
    int            best        = -VALUE_INF;
    if (!is_in_check)
    {
        best = evaluate(pos);
        if (best >= beta)
        {
            return best; // stand-pat cutoff — returns before the pin scan below
        }
        if (best > alpha)
        {
            alpha = best;
        }
    }

    Move           moves[MAX_MOVES];
    const int      count  = generate_pseudo(pos, moves, !is_in_check); // in check: evasions; else captures + promotions
    const Bitboard pinned = position_pinned_to_king(pos); // for the copy-free legality test in the move loop

    // MVV-LVA ordering.
    int scores[MAX_MOVES];
    for (int index = 0; index < count; index++)
    {
        const Move move  = moves[index];
        int        score = 0;
        if (move_is_capture(move))
        {
            score = 100 * SeeValue[move_is_ep(move) ? PAWN : pos->board[move_to(move)]] -
                    SeeValue[pos->board[move_from(move)]];
        }
        if (move_is_promo(move))
        {
            score += 900000 + move_promo_pt(move) - 1; // -1: keep the historic score values (promo_pt is 1-based +1)
        }
        scores[index] = score;
    }

    int legal_count = 0;
    for (int index = 0; index < count; index++)
    {
        int best_index = index;
        for (int other = index + 1; other < count; other++)
        {
            if (scores[other] > scores[best_index])
            {
                best_index = other;
            }
        }
        const Move swap_move = moves[index];
        moves[index]         = moves[best_index];
        moves[best_index]    = swap_move;
        const int swap_score = scores[index];
        scores[index]        = scores[best_index];
        scores[best_index]   = swap_score;
        const Move move      = moves[index];

        // Copy-free legality first, then SEE pruning, so only searched moves pay make_move.
        if (!position_is_legal_fast(pos, move, checkers, pinned))
        {
            continue;
        }
        legal_count++;

        if (!is_in_check && move_is_capture(move) && static_exchange_eval(pos, move) < 0)
        {
            continue; // skip losing captures
        }

        Position child = *pos;
        position_make_move(&child, move);
        tt_prefetch(child.key); // slot loads during the recursive-call setup
        const int score = -qsearch(searcher, &child, -beta, -alpha, ply + 1);
        if (g_stop)
        {
            return 0;
        }
        if (score > best)
        {
            best = score;
            if (score > alpha)
            {
                alpha = score;
                if (score >= beta)
                {
                    break;
                }
            }
        }
    }
    if (is_in_check && legal_count == 0)
    {
        return -VALUE_MATE + ply; // checkmate (all evasions were illegal)
    }
    return best;
}

/**
 * @brief Fail-soft PVS negamax: the main recursive search with all pruning, extensions, and move ordering.
 *
 * The `excluded` argument skips a move (for the singular-extension exclusion search), or is MOVE_NONE.
 */
static int negamax(Searcher *searcher, const Position *pos, int depth, int alpha, int beta, const int ply,
                   const bool is_cutnode, const Move prev_move, const Move excluded)
{
    if (is_time_up(searcher))
    {
        return 0; // is_time_up() already set g_stop for the main thread
    }
    const bool is_root    = ply == 0;
    const bool is_pv_node = beta - alpha > 1;
    searcher->pv_len[ply] = 0;

    if (!is_root)
    {
        if (is_draw(searcher, pos))
        {
            return draw_value();
        }
        // Mate-distance pruning.
        alpha = max_int(alpha, -VALUE_MATE + ply);
        beta  = min_int(beta, VALUE_MATE - ply - 1);
        if (alpha >= beta)
        {
            return alpha;
        }
    }
    if (ply >= MAX_PLY - 1)
    {
        return evaluate(pos);
    }
    if (depth <= 0)
    {
        return qsearch(searcher, pos, alpha, beta, ply);
    }

    searcher->nodes++;
    // Checkers once per node (drives IIR / reverse-futility / null-move / in-check logic below). The pinned
    // bitboard the legality test also needs is computed later, just before the move loop, so the frequent
    // TT / RFP / null-move cutoffs above it never pay for the pin scan.
    const Bitboard checkers    = position_attackers_to(pos, position_king_sq(pos, pos->color_to_move),
                                                       enemy_of(pos->color_to_move), position_occupied(pos));
    const bool     is_in_check = checkers != 0;

    // Continuation-history / countermove key = the (piece, to-square) of the move that reached this node.
    int prev_piece_to = -1;
    if (!move_is_none(prev_move))
    {
        const Piece prev_type = (Piece)pos->board[move_to(prev_move)];
        if (prev_type != NO_PIECE)
        {
            // History/countermove key: (color*6 + 0-based type) * 64 + to — the mover was the opponent.
            prev_piece_to = (enemy_of(pos->color_to_move) * 6 + (int)prev_type - 1) * 64 + move_to(prev_move);
        }
    }

    TTData     tt_entry  = {0};
    const bool is_tt_hit = tt_probe(pos->key, &tt_entry);
    const int  tt_score  = is_tt_hit ? score_from_tt((int)tt_entry.score, ply) : VALUE_NONE;
    const Move tt_move   = is_tt_hit ? (Move)tt_entry.move : MOVE_NONE;
    if (move_is_none(excluded) && !is_pv_node && is_tt_hit && (int)tt_entry.depth >= depth &&
        (tt_entry.bound == BOUND_EXACT || (tt_entry.bound == BOUND_LOWER && tt_score >= beta) ||
         (tt_entry.bound == BOUND_UPPER && tt_score <= alpha)))
    {
        return tt_score;
    }

    // Internal iterative reduction: with no TT move to anchor ordering at higher depths, search shallower
    // first so the cheaper search populates the TT move for the re-search.
    if (depth >= 4 && move_is_none(tt_move) && !is_in_check)
    {
        depth--;
    }

    // Raw static eval (stored in the TT); the corrected eval drives pruning/reductions. Keep them separate
    // so re-reading the TT eval never double-applies the correction.
    const int raw_eval =
        is_in_check ? VALUE_NONE : (is_tt_hit && tt_entry.eval != VALUE_NONE ? (int)tt_entry.eval : evaluate(pos));
    int eval = raw_eval;
    if (!is_in_check)
    {
        eval += searcher->correction_history[pos->color_to_move][pos->pawn_key & (CORRHIST_SIZE - 1)] / CORRHIST_GRAIN;
        eval = clamp_int(eval, -VALUE_MATE_IN_MAX + 1, VALUE_MATE_IN_MAX - 1);
    }

    // Reverse futility pruning (static null move).
    if (!is_pv_node && !is_in_check && depth <= 8 && !is_mate_score(beta) && eval - g_params.rfp_margin * depth >= beta)
    {
        return eval;
    }

    // Null-move pruning.
    if (!is_pv_node && !is_in_check && depth >= 3 && eval >= beta &&
        position_has_non_pawn_material(pos, pos->color_to_move))
    {
        const int reduction  = 3 + depth / 3 + min_int((eval - beta) / g_params.nmp_divisor, 3);
        Position  null_child = *pos;
        position_make_null(&null_child);
        searcher_hist_push(searcher, pos->key);
        const int score = -negamax(searcher, &null_child, depth - reduction, -beta, -beta + 1, ply + 1, !is_cutnode,
                                   MOVE_NONE, MOVE_NONE);
        searcher_hist_pop(searcher);
        if (g_stop)
        {
            return 0;
        }
        if (score >= beta)
        {
            return is_mate_score(score) ? beta : score;
        }
    }

    Move      moves[MAX_MOVES];
    const int count = generate_pseudo(pos, moves, false); // legality is filtered in the loop via a single make_move
    const Bitboard pinned = position_pinned_to_king(pos); // for the copy-free legality test in the move loop

    // Score moves: TT move, captures (MVV-LVA), killers, history.
    int scores[MAX_MOVES];
    for (int index = 0; index < count; index++)
    {
        Move move = moves[index];
        int  move_score;
        if (move == tt_move)
        {
            move_score = 2000000000;
        }
        else if (move_is_capture(move))
        {
            move_score = 1000000 + 100 * SeeValue[move_is_ep(move) ? PAWN : pos->board[move_to(move)]] -
                         SeeValue[pos->board[move_from(move)]];
        }
        else if (move_is_promo(move))
        {
            move_score = 900000 + move_promo_pt(move) - 1;
        }
        else if (move == searcher->killers[ply][0])
        {
            move_score = 800000;
        }
        else if (move == searcher->killers[ply][1])
        {
            move_score = 700000;
        }
        else if (prev_piece_to >= 0 && move == searcher->counter_moves[prev_piece_to])
        {
            move_score = 650000;
        }
        else
        {
            const int current_piece_to =
                (pos->color_to_move * 6 + pos->board[move_from(move)] - 1) * 64 + move_to(move);
            move_score = searcher->history[pos->color_to_move][move_from(move)][move_to(move)] +
                         (prev_piece_to >= 0 ? searcher->cont_hist[prev_piece_to * 768 + current_piece_to] : 0);
        }
        scores[index] = move_score;
    }

    int       best_score = -VALUE_INF;
    Move      best_move  = MOVE_NONE;
    const int orig_alpha = alpha;
    int       move_count = 0;
    Move      quiets[64];
    int       quiet_count = 0;

    for (int index = 0; index < count; index++)
    {
        int best_index = index;
        for (int other = index + 1; other < count; other++)
        {
            if (scores[other] > scores[best_index])
            {
                best_index = other;
            }
        }
        const Move swap_move = moves[index];
        moves[index]         = moves[best_index];
        moves[best_index]    = swap_move;
        const int swap_score = scores[index];
        scores[index]        = scores[best_index];
        scores[best_index]   = swap_score;
        const Move move      = moves[index];
        if (move == excluded)
        {
            continue; // singular exclusion search: skip the move being tested for singularity
        }

        // Copy-free legality: skip illegal pseudo-legal moves BEFORE any make_move, so the LMP/futility/SEE
        // pruning below runs first and only searched moves pay the ~2KB copy. Skipping before move_count++
        // keeps legal-move ordering/pruning identical to a legal generator.
        if (!position_is_legal_fast(pos, move, checkers, pinned))
        {
            continue;
        }
        const bool is_quiet = move_is_quiet(move);
        move_count++;

        // Late-move pruning: at low depth, stop trying quiet moves once deep into the ordered list.
        if (!is_pv_node && !is_in_check && is_quiet && depth <= 8 && move_count > g_params.lmp_base + depth * depth &&
            !is_mate_score(best_score))
        {
            continue;
        }

        // Futility pruning: at low depth, skip quiet moves that a margin cannot lift to alpha.
        if (!is_root && !is_pv_node && !is_in_check && is_quiet && depth <= 6 && move_count > 1 &&
            !is_mate_score(best_score) && eval + g_params.futility_base + g_params.futility_margin * depth <= alpha)
        {
            continue;
        }

        // SEE pruning of clearly-losing captures at low depth.
        if (!is_root && depth <= 6 && move_is_capture(move) && !is_mate_score(best_score) &&
            static_exchange_eval(pos, move) < -g_params.see_capture_margin * depth)
        {
            continue;
        }

        // Move survived pruning — make it now (legality already established above).
        Position child = *pos;
        position_make_move(&child, move);
        tt_prefetch(child.key); // slot loads while we finish this node before recursing into the child
        const bool is_child_in_check = position_is_in_check(&child);
        int        extension         = is_child_in_check ? 1 : 0;

        // Singular extension: if the TT move is much better than every alternative — an exclusion search
        // (this position without the TT move) at reduced depth fails low below a margin — extend it.
        if (!is_root && move == tt_move && move_is_none(excluded) && depth >= 8 && is_tt_hit &&
            (int)tt_entry.depth >= depth - 3 && (tt_entry.bound == BOUND_LOWER || tt_entry.bound == BOUND_EXACT) &&
            !is_mate_score(tt_score))
        {
            const int singular_beta  = tt_score - g_params.singular_margin * depth;
            const int singular_score = negamax(searcher, pos, (depth - 1) / 2, singular_beta - 1, singular_beta, ply,
                                               is_cutnode, prev_move, tt_move);
            if (singular_score < singular_beta)
            {
                extension = 1;
            }
        }

        const int new_depth = depth - 1 + extension;

        searcher_hist_push(searcher, pos->key);
        int score;
        if (move_count == 1)
        {
            score = -negamax(searcher, &child, new_depth, -beta, -alpha, ply + 1, false, move, MOVE_NONE);
        }
        else
        {
            int reduction = 0;
            if (depth >= 3 && move_count >= 4 && is_quiet && !is_in_check)
            {
                reduction = Reductions[min_int(depth, MAX_PLY - 1)][min_int(move_count, 63)];
                if (is_pv_node)
                {
                    reduction--;
                }
                if (is_cutnode)
                {
                    reduction++;
                }
                reduction = clamp_int(reduction, 0, new_depth - 1);
            }
            score =
                -negamax(searcher, &child, new_depth - reduction, -alpha - 1, -alpha, ply + 1, true, move, MOVE_NONE);
            if (score > alpha && reduction > 0)
            {
                score =
                    -negamax(searcher, &child, new_depth, -alpha - 1, -alpha, ply + 1, !is_cutnode, move, MOVE_NONE);
            }
            if (score > alpha && score < beta)
            {
                score = -negamax(searcher, &child, new_depth, -beta, -alpha, ply + 1, false, move, MOVE_NONE);
            }
        }
        searcher_hist_pop(searcher);
        if (g_stop)
        {
            return 0;
        }

        if (is_quiet && quiet_count < 64)
        {
            quiets[quiet_count++] = move;
        }

        if (score > best_score)
        {
            best_score = score;
            best_move  = move;
            if (score > alpha)
            {
                alpha = score;
                if (is_pv_node)
                {
                    update_pv(searcher, ply, move);
                }
                if (score >= beta)
                {
                    // Beta cutoff: reward the quiet cutoff move (main + continuation history, killer,
                    // countermove), punish the quiets that failed before it.
                    if (is_quiet)
                    {
                        if (searcher->killers[ply][0] != move)
                        {
                            searcher->killers[ply][1] = searcher->killers[ply][0];
                            searcher->killers[ply][0] = move;
                        }
                        if (prev_piece_to >= 0)
                        {
                            searcher->counter_moves[prev_piece_to] = move;
                        }
                        const int bonus = min_int(depth * depth, g_params.history_max);
                        const int current_piece_to =
                            (pos->color_to_move * 6 + pos->board[move_from(move)] - 1) * 64 + move_to(move);
                        apply_gravity(&searcher->history[pos->color_to_move][move_from(move)][move_to(move)], bonus);
                        if (prev_piece_to >= 0)
                        {
                            apply_gravity(&searcher->cont_hist[prev_piece_to * 768 + current_piece_to], bonus);
                        }
                        for (int quiet_index = 0; quiet_index < quiet_count - 1; quiet_index++)
                        {
                            const Move quiet_move = quiets[quiet_index];
                            apply_gravity(
                                &searcher->history[pos->color_to_move][move_from(quiet_move)][move_to(quiet_move)],
                                -bonus);
                            if (prev_piece_to >= 0)
                            {
                                apply_gravity(
                                    &searcher
                                         ->cont_hist[prev_piece_to * 768 +
                                                     (pos->color_to_move * 6 + pos->board[move_from(quiet_move)] - 1) *
                                                         64 +
                                                     move_to(quiet_move)],
                                    -bonus);
                            }
                        }
                    }
                    break;
                }
            }
        }
    }

    if (move_count == 0)
    {
        return is_in_check ? -VALUE_MATE + ply : draw_value(); // no legal move: checkmate or stalemate
    }

    const Bound bound = best_score >= beta ? BOUND_LOWER : (alpha > orig_alpha ? BOUND_EXACT : BOUND_UPPER);
    if (move_is_none(excluded))
    {
        tt_store(pos->key, best_score, is_in_check ? VALUE_NONE : raw_eval, depth, bound, best_move, ply);

        // Update the eval correction: blend in (search score - raw static eval), but only when the score is
        // a trustworthy signal — not in check, not a tactical (capture) best move, not a mate, and the bound
        // does not contradict the direction of the correction.
        if (!is_in_check && !is_mate_score(best_score) && (move_is_none(best_move) || !move_is_capture(best_move)) &&
            !(bound == BOUND_LOWER && best_score <= raw_eval) && !(bound == BOUND_UPPER && best_score >= raw_eval))
        {
            int *const entry  = &searcher->correction_history[pos->color_to_move][pos->pawn_key & (CORRHIST_SIZE - 1)];
            const int  target = clamp_int((best_score - raw_eval) * CORRHIST_GRAIN, -CORRHIST_MAX, CORRHIST_MAX);
            const int  weight = min_int(depth + 1, 16);
            *entry = clamp_int((*entry * (256 - weight) + target * weight) / 256, -CORRHIST_MAX, CORRHIST_MAX);
        }
    }
    if (is_root)
    {
        searcher->root_best = best_move;
    }
    return best_score;
}

Move searcher_go(Searcher *searcher, Position root, const SearchLimits *lim, const bool is_main_thread)
{
    searcher->is_main = is_main_thread;
    if (searcher->is_main)
    {
        g_stop = false; // clear the shared stop before a new search (helpers are launched after this)
    }
    searcher->nodes    = 0;
    searcher->seldepth = 0;
    memset(searcher->killers, 0, sizeof(searcher->killers));
    memset(searcher->history, 0, sizeof(searcher->history));
    memset(searcher->correction_history, 0, sizeof(searcher->correction_history));
    memset(searcher->counter_moves, 0, sizeof(searcher->counter_moves));
    memset(searcher->cont_hist, 0, sizeof(searcher->cont_hist));
    searcher->start_ms = platform_now_ms();
    set_time(searcher, &root, lim);
    if (searcher->is_main)
    {
        tt_new_search(); // bump generation once per search, not per helper thread
    }
    if (nnue_is_loaded())
    {
        nnue_refresh(&root.accumulator, &root); // authoritative root accumulator (robust to a net loaded mid-game)
    }

    searcher->root_best = MOVE_NONE;
    Move      best      = MOVE_NONE;
    const int max_depth = lim->depth > 0 ? lim->depth : MAX_PLY - 2;
    int       score     = 0;

    for (int depth = 1; depth <= max_depth; depth++)
    {
        // Aspiration windows once we have a score to trust.
        int alpha = -VALUE_INF, beta = VALUE_INF, delta = g_params.aspiration_delta;
        if (depth >= 4)
        {
            alpha = max_int(-VALUE_INF, score - delta);
            beta  = min_int(VALUE_INF, score + delta);
        }
        while (true)
        {
            const int window_score = negamax(searcher, &root, depth, alpha, beta, 0, false, MOVE_NONE, MOVE_NONE);
            if (g_stop)
            {
                break;
            }
            score = window_score;
            if (window_score <= alpha)
            {
                beta  = (alpha + beta) / 2;
                alpha = max_int(-VALUE_INF, window_score - delta);
                delta += delta / 2;
            }
            else if (window_score >= beta)
            {
                beta = min_int(VALUE_INF, window_score + delta);
                delta += delta / 2;
            }
            else
            {
                break;
            }
        }
        if (g_stop && best != MOVE_NONE)
        {
            break;
        }
        best = searcher->root_best;

        if (searcher->is_main && !searcher->is_silent)
        {
            const int64_t  elapsed_ms       = elapsed(searcher);
            const uint64_t nodes_per_second = elapsed_ms ? searcher->nodes * 1000 / elapsed_ms : searcher->nodes;
            // Score string.
            char score_str[32];
            if (is_mate_score(score))
            {
                const int mate = score > 0 ? (VALUE_MATE - score + 1) / 2 : -(VALUE_MATE + score) / 2;
                snprintf(score_str, sizeof score_str, "mate %d", mate);
            }
            else
            {
                snprintf(score_str, sizeof score_str, "cp %d", score);
            }
            printf("info depth %d seldepth %d score %s nodes %llu nps %llu time %lld hashfull %d pv", depth,
                   searcher->seldepth, score_str, (unsigned long long)searcher->nodes,
                   (unsigned long long)nodes_per_second, (long long)elapsed_ms, tt_hashfull());
            for (int pv_index = 0; pv_index < searcher->pv_len[0]; pv_index++)
            {
                char uci_buf[8];
                printf(" %s", move_to_uci(searcher->pv_table[0][pv_index], uci_buf));
            }
            printf("\n");
            fflush(stdout);
        }

        if (g_stop)
        {
            break;
        }
        // Only the main thread stops on the time budget; helpers keep deepening to fill the shared TT
        // until the main thread ends the search.
        if (searcher->is_main && searcher->is_time_limited && elapsed(searcher) >= searcher->soft_ms)
        {
            break; // don't start a deeper iteration we can't finish
        }
        if (searcher->is_main && is_mate_score(score) && lim->depth == 0 && !lim->is_infinite)
        {
            break;
        }
    }
    if (searcher->is_main)
    {
        g_stop = true; // release the helper threads
    }
    searcher->root_score = score;
    return move_is_none(best) ? searcher->root_best : best;
}
