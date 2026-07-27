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
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ln(n) in Q28 fixed point for the LMR table (generated; see tools/generate_tables.py).
#include "generated/ln_tables.inc"

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

/**
 * @brief (Re)build @p shared's LMR reduction table from its current LmrBase/LmrDivisor parameters.
 *
 * Pure 64-bit fixed-point integer arithmetic (Q28), with ln(n) taken from the generated LnQ28 table — no
 * floating point anywhere, so the table (and with it the bench node signature) is bit-identical on every
 * platform and compiler by construction. Q28 reproduces the previous double computation exactly: its
 * ~4e-9 error is four orders of magnitude inside the closest truncation boundary (~8e-5) over the whole
 * parameter grid, and the largest intermediate (ln(127)^2 in Q56) uses 61 of the 63 available bits.
 */
static void build_reductions(SearchShared *shared)
{
    // reduction = floor(lmr_base/100 + ln(depth) * ln(move_number) / (lmr_divisor/100)), all in Q28.
    const int64_t base_q28 = ((int64_t)shared->params.lmr_base_x100 << 28) / 100;
    for (int depth = 1; depth < MAX_PLY; depth++)
    {
        for (int move_number = 1; move_number < 64; move_number++)
        {
            const int64_t product_q28              = (LnQ28[depth] * LnQ28[move_number]) >> 28;
            const int64_t term_q28                 = product_q28 * 100 / shared->params.lmr_divisor_x100;
            shared->reductions[depth][move_number] = (int)((base_q28 + term_q28) >> 28);
        }
    }
}

void search_shared_init(SearchShared *shared)
{
    memset(shared, 0, sizeof *shared);
    shared->params = (SearchParams){
        .rfp_margin         = 56,
        .nmp_divisor        = 199,
        .lmp_base           = 5,
        .futility_base      = 94,
        .futility_margin    = 98,
        .see_capture_margin = 97,
        .lmr_base_x100      = 90,
        .lmr_divisor_x100   = 220,
        .singular_margin    = 3,
        .aspiration_delta   = 21,
        .history_max        = 402,
    };
    build_reductions(shared);
}

bool set_search_param(SearchShared *shared, const char *name, const int value)
{
    // One row per tunable: its UCI name and the field it writes. The two LMR params also rebuild the
    // precomputed reduction table (init_search reads them), flagged by rebuilds_lmr_table.
    static const struct
    {
        const char *name;
        size_t      offset; ///< field offset in SearchParams (keeps the table static const)
        bool        rebuilds_lmr_table;
    } params[] = {
        {"RfpMargin", offsetof(SearchParams, rfp_margin), false},
        {"NmpDivisor", offsetof(SearchParams, nmp_divisor), false},
        {"LmpBase", offsetof(SearchParams, lmp_base), false},
        {"FutilityBase", offsetof(SearchParams, futility_base), false},
        {"FutilityMargin", offsetof(SearchParams, futility_margin), false},
        {"SeeCaptureMargin", offsetof(SearchParams, see_capture_margin), false},
        {"SingularMargin", offsetof(SearchParams, singular_margin), false},
        {"AspirationDelta", offsetof(SearchParams, aspiration_delta), false},
        {"HistoryMax", offsetof(SearchParams, history_max), false},
        {"LmrBase", offsetof(SearchParams, lmr_base_x100), true},
        {"LmrDivisor", offsetof(SearchParams, lmr_divisor_x100), true},
    };

    for (size_t i = 0; i < sizeof params / sizeof params[0]; i++)
    {
        if (strcmp(name, params[i].name) == 0)
        {
            *(int *)((char *)&shared->params + params[i].offset) = value;
            if (params[i].rebuilds_lmr_table)
            {
                build_reductions(shared);
            }
            return true;
        }
    }
    return false;
}

static int64_t elapsed(const Searcher *searcher)
{
    return platform_now_ms() - searcher->start_ms;
}

/** @brief Whether the search must stop now (shared stop set, or the main thread hit its node/time budget). */
static bool is_time_up(const Searcher *searcher)
{
    if (atomic_load_explicit(&searcher->engine->search.is_stop_requested, memory_order_relaxed))
    {
        return true;
    }
    // Only the main thread owns the time/node budget; when it runs out it sets the shared stop so helpers stop too.
    if (searcher->is_main && ((searcher->node_limit && searcher->nodes >= (uint64_t)searcher->node_limit) ||
                              (searcher->is_time_limited && elapsed(searcher) >= searcher->hard_ms)))
    {
        atomic_store_explicit(&searcher->engine->search.is_stop_requested, true, memory_order_relaxed);
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
        return 0; // is_time_up() already set the shared stop for the main thread
    }
    searcher->nodes++;
    if (ply > searcher->seldepth)
    {
        searcher->seldepth = ply;
    }
    if (ply >= MAX_PLY - 1)
    {
        return evaluate(pos, &searcher->engine->eval_cache);
    }

    const Bitboard checkers    = pos->checkers; // cached: recomputed once per make_move
    const bool     is_in_check = checkers != 0;

    // Transposition table: qsearch entries are stored at depth 0, so ANY hit covers a qsearch node's needs
    // (the tree below is captures-only and the stored score already summarizes it). Non-PV nodes take the
    // usual bound cutoffs; the entry's static eval also seeds stand-pat below without re-evaluating.
    TTData     tt_entry  = {0};
    const bool is_tt_hit = tt_probe(&searcher->engine->tt, pos->key, &tt_entry);
    if (is_tt_hit && beta - alpha == 1)
    {
        const int tt_score = score_from_tt((int)tt_entry.score, ply);
        if (tt_entry.bound == BOUND_EXACT || (tt_entry.bound == BOUND_LOWER && tt_score >= beta) ||
            (tt_entry.bound == BOUND_UPPER && tt_score <= alpha))
        {
            return tt_score;
        }
    }

    const int orig_alpha = alpha;
    int       best       = -VALUE_INF;
    if (!is_in_check)
    {
        best = (is_tt_hit && tt_entry.eval != VALUE_NONE) ? (int)tt_entry.eval
                                                          : evaluate(pos, &searcher->engine->eval_cache);
        if (best >= beta)
        {
            return best; // stand-pat cutoff — returns before the pin scan below
        }
        if (best > alpha)
        {
            alpha = best;
        }
    }
    const int stand_pat = best; // raw static eval (or VALUE_NONE-ish -INF in check) for the TT store below

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
        if (!position_is_legal(pos, move, checkers, pinned))
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
        tt_prefetch(&searcher->engine->tt, child.key); // slot loads during the recursive-call setup
        const int score = -qsearch(searcher, &child, -beta, -alpha, ply + 1);
        if (atomic_load_explicit(&searcher->engine->search.is_stop_requested, memory_order_relaxed))
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
    // Store at depth 0 so negamax's depth-preferred probes still outrank this entry; the raw stand-pat eval
    // rides along for other nodes' stand-pat seeding. Bound classification mirrors negamax's fail-soft rules.
    const Bound bound = best >= beta ? BOUND_LOWER : (alpha > orig_alpha ? BOUND_EXACT : BOUND_UPPER);
    tt_store(&searcher->engine->tt, pos->key, best, is_in_check ? VALUE_NONE : stand_pat, 0, bound, MOVE_NONE, ply);
    return best;
}

/**
 * @brief Fail-soft PVS negamax: the main recursive search with all pruning, extensions, and move ordering.
 *
 * Negamax: scores are always from the side to move's point of view, so one function serves both colors —
 * a child's score is negated (`-negamax(...)`) to convert it to the parent's perspective. Alpha-beta: the
 * window (alpha, beta) brackets the scores that can still influence the result; a score >= beta refutes the
 * opponent's previous choice ("beta cutoff" — the rest of the moves need not be searched), a score <= alpha
 * cannot improve on an alternative already available higher in the tree. Fail-soft: the returned score may
 * lie outside the window (best_score is tracked independently of alpha), which gives the TT tighter bounds.
 * PVS (principal variation search): only the first move gets the full window; later moves get a cheap
 * null-window "is it better than alpha at all?" test and are re-searched with a wider window only when that
 * test fails high — see the move loop.
 *
 * @param searcher per-thread search state (histories, killers, PV, node counts).
 * @param pos the position at this node (copy-make: children are copies, there is no unmake).
 * @param depth remaining full-width depth in plies; <= 0 drops into qsearch.
 * @param alpha lower bound of the search window (side-to-move POV).
 * @param beta upper bound of the search window.
 * @param ply distance from the root (drives mate scoring and per-ply tables).
 * @param is_cutnode whether this node is *expected* to fail high (>= beta) — an alpha-beta tree-shape
 *        prediction used only to bias LMR; a wrong prediction costs a re-search, never correctness.
 * @param prev_move the move that led to this node (keys the countermove/continuation-history tables), or
 *        MOVE_NONE after a null move / at the root.
 * @param excluded a move to pretend does not exist, for the singular-extension exclusion search ("how good
 *        is this node WITHOUT the TT move?"), or MOVE_NONE for a normal search.
 * @return the best score found (side-to-move POV); mates are encoded as VALUE_MATE - ply.
 */
static int negamax(Searcher *searcher, const Position *pos, int depth, int alpha, int beta, const int ply,
                   const bool is_cutnode, const Move prev_move, const Move excluded)
{
    if (is_time_up(searcher))
    {
        return 0; // is_time_up() already set the shared stop for the main thread
    }
    const bool is_root = ply == 0;
    // A PV node was given a real window by its parent (first move, or a re-search); everything searched with
    // a null window (beta == alpha+1) is a yes/no test node where pruning can be more aggressive.
    const bool is_pv_node = beta - alpha > 1;
    searcher->pv_len[ply] = 0;

    if (!is_root)
    {
        if (position_is_draw(pos, searcher->hist_keys, searcher->hist_count))
        {
            return draw_value(); // repetition / 50-move / insufficient material
        }
        // Mate-distance pruning: at ply p the best possible outcome is mate-in-p (VALUE_MATE - p) and the
        // worst is being mated now (-VALUE_MATE + p). Clamping the window to those limits fails immediately
        // when a shorter mate is already known higher in the tree — no point searching for a longer one.
        alpha = max_int(alpha, -VALUE_MATE + ply);
        beta  = min_int(beta, VALUE_MATE - ply - 1);
        if (alpha >= beta)
        {
            return alpha;
        }
    }
    if (ply >= MAX_PLY - 1)
    {
        return evaluate(
            pos,
            &searcher->engine->eval_cache); // out of ply stack — return the static eval rather than recursing further
    }
    if (depth <= 0)
    {
        // Horizon reached: resolve captures/checks with quiescence search instead of returning a raw eval,
        // so the score is not blind to a hanging queen on the last searched move.
        return qsearch(searcher, pos, alpha, beta, ply);
    }

    searcher->nodes++;
    // Checkers drive the IIR / reverse-futility / null-move / in-check logic below; read the cached set
    // (make_move computed it for this node). The pinned bitboard the legality test also needs is computed
    // later, just before the move loop, so the frequent TT / RFP / null-move cutoffs never pay for the pin scan.
    const Bitboard checkers    = pos->checkers;
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

    // Transposition table probe. A stored entry carries a score bound (EXACT / LOWER = failed high /
    // UPPER = failed low) from a search of at least `tt_entry.depth`; when that depth covers ours and the
    // bound is decisive against the current window, return it without searching. Cutoffs are skipped in PV
    // nodes (they must produce a full PV line, not a bound) and during a singular exclusion search (the TT
    // entry describes the position WITH the excluded move). Even without a cutoff, the entry contributes its
    // move (ordering anchor) and static eval.
    TTData     tt_entry  = {0};
    const bool is_tt_hit = tt_probe(&searcher->engine->tt, pos->key, &tt_entry);
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
    // so re-reading the TT eval never double-applies the correction. The correction history is a running
    // average of (search result - static eval) keyed by pawn structure — it nudges the eval toward what
    // deeper searches of similar structures actually returned. In check there is no meaningful static eval.
    const int raw_eval =
        is_in_check ? VALUE_NONE
                    : (is_tt_hit && tt_entry.eval != VALUE_NONE ? (int)tt_entry.eval
                                                                : evaluate(pos, &searcher->engine->eval_cache));
    int eval = raw_eval;
    if (!is_in_check)
    {
        eval += searcher->correction_history[pos->color_to_move][pos->pawn_key & (CORRHIST_SIZE - 1)] / CORRHIST_GRAIN;
        eval = clamp_int(eval, -VALUE_MATE_IN_MAX + 1, VALUE_MATE_IN_MAX - 1);
    }

    // Reverse futility pruning (static null move): if the static eval beats beta by a depth-scaled safety
    // margin, the opponent's previous move was almost certainly a blunder we don't need a search to refute —
    // fail high on the eval alone. Shallow depths only (the margin models how much can change per ply).
    if (!is_pv_node && !is_in_check && depth <= 8 && !is_mate_score(beta) &&
        eval - searcher->engine->search.params.rfp_margin * depth >= beta)
    {
        return eval;
    }

    // Null-move pruning: give the opponent a free extra move (we "pass"); if a reduced-depth search STILL
    // fails high, the position is so good that a real move can only be better — fail high without searching
    // our moves. The reduction grows with depth and with how far eval already exceeds beta. Unsound in
    // zugzwang, where passing is the best "move" — hence the non-pawn-material guard (zugzwang is essentially
    // a pawn/king-endgame phenomenon), and unproven mate scores from the reduced search are clamped to beta.
    if (!is_pv_node && !is_in_check && depth >= 3 && eval >= beta &&
        position_has_non_pawn_material(pos, pos->color_to_move))
    {
        const int reduction  = 3 + depth / 3 + min_int((eval - beta) / searcher->engine->search.params.nmp_divisor, 3);
        Position  null_child = *pos;
        position_make_null(&null_child); // flips side to move (and clears ep); board unchanged
        searcher_hist_push(searcher, pos->key);
        const int score = -negamax(searcher, &null_child, depth - reduction, -beta, -beta + 1, ply + 1, !is_cutnode,
                                   MOVE_NONE, MOVE_NONE);
        searcher_hist_pop(searcher);
        if (atomic_load_explicit(&searcher->engine->search.is_stop_requested, memory_order_relaxed))
        {
            return 0;
        }
        if (score >= beta)
        {
            return is_mate_score(score) ? beta : score;
        }
    }

    Move           moves[MAX_MOVES];
    const int      count  = generate_pseudo(pos, moves, false); // legality is filtered in the loop, copy-free
    const Bitboard pinned = position_pinned_to_king(pos);       // for the copy-free legality test in the move loop

    // Score every move for ordering. Good ordering is what makes alpha-beta cut: searching the best move
    // first makes every later move a cheap null-window refutation. Tiers, best first:
    //   2000000000  TT move — the best move of a previous (usually shallower) search of this node
    //      1000000+ captures by MVV-LVA (most valuable victim first, least valuable attacker as tiebreak)
    //       900000+ non-capture promotions (queen first)
    //   800k / 700k killers — quiets that caused a beta cutoff at this ply in a sibling subtree
    //       650000  countermove — the quiet that historically refutes prev_move specifically
    //         else  history: butterfly [side][from][to] plus 1-ply continuation history, both fed by
    //               beta-cutoff bonuses/penalties below (typical magnitude well under the named tiers)
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
    const int orig_alpha = alpha; // remembered to classify the final TT bound (did anything raise alpha?)
    int       move_count = 0;     // LEGAL moves searched so far (drives LMP/LMR move-count rules)
    Move      quiets[64];         // quiets already searched, so a later cutoff can penalize their history
    int       quiet_count = 0;

    for (int index = 0; index < count; index++)
    {
        // Incremental selection sort: swap the highest-scored remaining move to the front. Sorting lazily —
        // one selection per iteration instead of a full sort up front — means a node that cuts off after a
        // few moves (the common case) never pays to order the rest.
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
        if (!position_is_legal(pos, move, checkers, pinned))
        {
            continue;
        }
        const bool is_quiet = move_is_quiet(move);
        move_count++;

        // Late-move pruning: at low depth, once this deep into a well-ordered move list, the remaining quiet
        // moves are so unlikely to be best that they are skipped outright (the quota grows with depth^2).
        if (!is_pv_node && !is_in_check && is_quiet && depth <= 8 &&
            move_count > searcher->engine->search.params.lmp_base + depth * depth && !is_mate_score(best_score))
        {
            continue;
        }

        // Futility pruning: at low depth, if the static eval plus an optimistic depth-scaled margin still
        // cannot reach alpha, a quiet move has no realistic way to matter — skip it without searching.
        if (!is_root && !is_pv_node && !is_in_check && is_quiet && depth <= 6 && move_count > 1 &&
            !is_mate_score(best_score) &&
            eval + searcher->engine->search.params.futility_base +
                    searcher->engine->search.params.futility_margin * depth <=
                alpha)
        {
            continue;
        }

        // SEE pruning: skip captures that static exchange evaluation shows lose material beyond a
        // depth-scaled tolerance (deeper search keeps more marginal captures for tactical safety).
        if (!is_root && depth <= 6 && move_is_capture(move) && !is_mate_score(best_score) &&
            static_exchange_eval(pos, move) < -searcher->engine->search.params.see_capture_margin * depth)
        {
            continue;
        }

        // Move survived pruning — make it now (legality already established above).
        Position child = *pos;
        position_make_move(&child, move);
        tt_prefetch(&searcher->engine->tt,
                    child.key); // slot loads while we finish this node before recursing into the child

        // Check extension: search checking moves one ply deeper — forcing sequences resolve instead of
        // being pushed past the horizon, and evasions are never the last searched ply.
        const bool is_child_in_check = position_is_in_check(&child);
        int        extension         = is_child_in_check ? 1 : 0;

        // Singular extension: if the TT move is much better than every alternative, this node hinges on one
        // move and deserves a deeper look. Test by re-searching this SAME position at reduced depth with the
        // TT move excluded, against a window just below the TT score: if even the best alternative fails low
        // of that margin, the TT move is "singular" — extend it. Requires a trustworthy TT entry (adequate
        // depth, lower/exact bound); skipped inside an exclusion search (no recursive singularity testing).
        if (!is_root && move == tt_move && move_is_none(excluded) && depth >= 8 && is_tt_hit &&
            (int)tt_entry.depth >= depth - 3 && (tt_entry.bound == BOUND_LOWER || tt_entry.bound == BOUND_EXACT) &&
            !is_mate_score(tt_score))
        {
            const int singular_beta  = tt_score - searcher->engine->search.params.singular_margin * depth;
            const int singular_score = negamax(searcher, pos, (depth - 1) / 2, singular_beta - 1, singular_beta, ply,
                                               is_cutnode, prev_move, tt_move);
            if (singular_score < singular_beta)
            {
                extension = 1;
            }
        }

        const int new_depth = depth - 1 + extension;

        // The current key joins the repetition history for the child's is_draw scans (copy-make has no
        // undo stack, so this push/pop pair is what threads the game line through the recursion).
        searcher_hist_push(searcher, pos->key);
        int score;
        if (move_count == 1)
        {
            // PVS stage 1: the first (best-ordered) move gets the full (alpha, beta) window — it is expected
            // to become the PV move, and its exact score is what the null-window tests below compare against.
            score = -negamax(searcher, &child, new_depth, -beta, -alpha, ply + 1, false, move, MOVE_NONE);
        }
        else
        {
            // Late move reductions: with good ordering, late quiet moves almost never beat the first move,
            // so probe them at a reduced depth first. The base reduction grows ~log(depth) * log(move_count)
            // (the precomputed table), searched a bit deeper in PV nodes and a bit shallower at expected
            // cutnodes. A reduced search that beats alpha is re-searched at full depth below — a reduction
            // can cost a re-search but never a missed best move.
            int reduction = 0;
            if (depth >= 3 && move_count >= 4 && is_quiet && !is_in_check)
            {
                reduction = searcher->engine->search.reductions[min_int(depth, MAX_PLY - 1)][min_int(move_count, 63)];
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
            // PVS stage 2: null-window (alpha, alpha+1) scout at the (possibly reduced) depth — a cheap
            // yes/no "can this move beat alpha?". Most moves answer no and are done.
            score =
                -negamax(searcher, &child, new_depth - reduction, -alpha - 1, -alpha, ply + 1, true, move, MOVE_NONE);
            if (score > alpha && reduction > 0)
            {
                // The reduced scout failed high: re-run the scout at full depth before trusting it.
                score =
                    -negamax(searcher, &child, new_depth, -alpha - 1, -alpha, ply + 1, !is_cutnode, move, MOVE_NONE);
            }
            if (score > alpha && score < beta)
            {
                // PVS stage 3: the move genuinely beats alpha inside an open window — it is a new PV
                // candidate, so re-search with the full window for its exact score.
                score = -negamax(searcher, &child, new_depth, -beta, -alpha, ply + 1, false, move, MOVE_NONE);
            }
        }
        searcher_hist_pop(searcher);
        if (atomic_load_explicit(&searcher->engine->search.is_stop_requested, memory_order_relaxed))
        {
            return 0; // aborted mid-move: the partial score is garbage; iterative deepening keeps the last full result
        }

        if (is_quiet && quiet_count < 64)
        {
            quiets[quiet_count++] = move; // remember for the history penalty applied on a later beta cutoff
        }

        // Score bookkeeping: best_score tracks the true best (fail-soft — it may stay below alpha); alpha
        // rises only when a move beats it, and a score reaching beta refutes the opponent's previous move,
        // ending the node (beta cutoff).
        if (score > best_score)
        {
            best_score = score;
            best_move  = move;
            if (score > alpha)
            {
                alpha = score;
                if (is_pv_node)
                {
                    update_pv(searcher, ply, move); // extend this ply's PV line with the new best move
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
                        const int bonus = min_int(depth * depth, searcher->engine->search.params.history_max);
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
        // No legal move: in check it is checkmate — scored as "mated in ply" so nearer mates score worse
        // (and wins prefer the shortest mate) — otherwise stalemate.
        return is_in_check ? -VALUE_MATE + ply : draw_value();
    }

    // Classify the result for the TT: a score at/above beta only proves a lower bound (the search stopped
    // early at the cutoff); if nothing raised alpha, best_score is only an upper bound (every move might be
    // even worse than reported under fail-soft); in between, the score is exact. Nothing is stored from an
    // exclusion search — those scores describe an artificial position (a move pretended away).
    const Bound bound = best_score >= beta ? BOUND_LOWER : (alpha > orig_alpha ? BOUND_EXACT : BOUND_UPPER);
    if (move_is_none(excluded))
    {
        tt_store(&searcher->engine->tt, pos->key, best_score, is_in_check ? VALUE_NONE : raw_eval, depth, bound,
                 best_move, ply);

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
        atomic_store_explicit(&searcher->engine->search.is_stop_requested, false, memory_order_relaxed);
        // ^ clear the shared stop before a new search (helpers are launched after this)
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
        tt_new_search(&searcher->engine->tt); // bump generation once per search, not per helper thread
    }
    root.accumulator.cache = &searcher->refresh_cache; // this thread's finny cache, inherited by all children
    if (root.accumulator.net != NULL)
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
        int alpha = -VALUE_INF, beta = VALUE_INF, delta = searcher->engine->search.params.aspiration_delta;
        if (depth >= 4)
        {
            alpha = max_int(-VALUE_INF, score - delta);
            beta  = min_int(VALUE_INF, score + delta);
        }
        while (true)
        {
            const int window_score = negamax(searcher, &root, depth, alpha, beta, 0, false, MOVE_NONE, MOVE_NONE);
            if (atomic_load_explicit(&searcher->engine->search.is_stop_requested, memory_order_relaxed))
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
        if (atomic_load_explicit(&searcher->engine->search.is_stop_requested, memory_order_relaxed) &&
            best != MOVE_NONE)
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
                   (unsigned long long)nodes_per_second, (long long)elapsed_ms, tt_hashfull(&searcher->engine->tt));
            for (int pv_index = 0; pv_index < searcher->pv_len[0]; pv_index++)
            {
                char uci_buf[8];
                printf(" %s", move_to_uci(searcher->pv_table[0][pv_index], uci_buf));
            }
            printf("\n");
            fflush(stdout);
        }

        if (atomic_load_explicit(&searcher->engine->search.is_stop_requested, memory_order_relaxed))
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
        atomic_store_explicit(&searcher->engine->search.is_stop_requested, true,
                              memory_order_relaxed); // release the helpers
    }
    searcher->root_score = score;
    return move_is_none(best) ? searcher->root_best : best;
}
