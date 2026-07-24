#include "search.h"
#include "bitboard.h"
#include "eval.h"
#include "nnue.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

std::atomic<bool> g_stop{false};
SearchParams      g_params;

namespace
{

int       Reductions[MAX_PLY][64];
const int SeeValue[6] = {100, 320, 330, 500, 900, 10000};

// Eval correction history: a running average of (search score - static eval) keyed by pawn structure, used
// to nudge the static eval toward what search has historically found. Entries are cp * CORRHIST_GRAIN.
constexpr int CORRHIST_SIZE  = 16384; // power of two -> index by (pawn_key & (SIZE-1))
constexpr int CORRHIST_GRAIN = 256;
constexpr int CORRHIST_MAX   = 64 * CORRHIST_GRAIN; // clamp the correction to +/-64 cp

int draw_value()
{
    return 0;
}

// Static Exchange Evaluation of a capture: net material after the optimal capture sequence on move.to().
int static_exchange_eval(const Position &pos, Move move)
{
    int to = move.to(), from = move.from();
    int captured = move.is_ep() ? SeeValue[PAWN] : SeeValue[type_of(pos.board[to])];
    int gain[32];
    int swap_index = 0;
    gain[0]        = captured;

    Bitboard occupied = pos.occupied();
    if (move.is_ep())
    {
        occupied ^= sq_bb(to + (pos.stm == WHITE ? -8 : 8));
    }
    occupied ^= sq_bb(from);

    PieceType attacker = type_of(pos.board[from]);
    Color     side     = ~pos.stm;
    // Attackers hitting `to` given the current occupancy; recomputed each ply so x-rays reveal.
    Bitboard attackers = (pos.attackers_to(to, WHITE, occupied) | pos.attackers_to(to, BLACK, occupied)) & occupied;

    while (true)
    {
        swap_index++;
        gain[swap_index]        = SeeValue[attacker] - gain[swap_index - 1];
        Bitboard side_attackers = attackers & pos.by_color[side];
        if (!side_attackers)
        {
            break;
        }
        // Least valuable attacker of `side`.
        PieceType least_valuable_attacker = KING;
        Bitboard  attacker_bit            = 0;
        for (int piece_type = PAWN; piece_type <= KING; piece_type++)
        {
            Bitboard candidates = side_attackers & pos.pieces(side, PieceType(piece_type));
            if (candidates)
            {
                least_valuable_attacker = PieceType(piece_type);
                attacker_bit            = candidates & (~candidates + 1);
                break;
            }
        }
        attacker = least_valuable_attacker;
        occupied ^= attacker_bit;
        attackers = (pos.attackers_to(to, WHITE, occupied) | pos.attackers_to(to, BLACK, occupied)) & occupied;
        side      = ~side;
        if (attacker == KING && (attackers & pos.by_color[side]))
        {
            // Cannot recapture with the king into a still-defended square; stop before it.
            break;
        }
    }
    while (--swap_index > 0)
    {
        gain[swap_index - 1] = -std::max(-gain[swap_index - 1], gain[swap_index]);
    }
    return gain[0];
}

} // namespace

void init_search()
{
    double lmr_base    = g_params.lmr_base_x100 / 100.0;
    double lmr_divisor = g_params.lmr_divisor_x100 / 100.0;
    for (int depth = 1; depth < MAX_PLY; depth++)
    {
        for (int move_number = 1; move_number < 64; move_number++)
        {
            Reductions[depth][move_number] = int(lmr_base + std::log(depth) * std::log(move_number) / lmr_divisor);
        }
    }
}

bool set_search_param(const std::string &name, int value)
{
    if (name == "RfpMargin")
    {
        g_params.rfp_margin = value;
    }
    else if (name == "NmpDivisor")
    {
        g_params.nmp_divisor = value;
    }
    else if (name == "LmpBase")
    {
        g_params.lmp_base = value;
    }
    else if (name == "FutilityBase")
    {
        g_params.futility_base = value;
    }
    else if (name == "FutilityMargin")
    {
        g_params.futility_margin = value;
    }
    else if (name == "SeeCaptureMargin")
    {
        g_params.see_capture_margin = value;
    }
    else if (name == "SingularMargin")
    {
        g_params.singular_margin = value;
    }
    else if (name == "AspirationDelta")
    {
        g_params.aspiration_delta = value;
    }
    else if (name == "HistoryMax")
    {
        g_params.history_max = value;
    }
    else if (name == "LmrBase")
    {
        g_params.lmr_base_x100 = value;
        init_search(); // LMR table depends on this
    }
    else if (name == "LmrDivisor")
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

int64_t Searcher::elapsed() const
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
}

bool Searcher::time_up()
{
    if (g_stop.load(std::memory_order_relaxed))
    {
        return true;
    }
    // Only the main thread owns the time/node budget; when it runs out it sets g_stop so helpers stop too.
    if (is_main && ((node_limit && nodes >= (uint64_t)node_limit) || (use_time && elapsed() >= hard_ms)))
    {
        g_stop.store(true, std::memory_order_relaxed);
        return true;
    }
    return false;
}

void Searcher::set_time(const Position &root, const SearchLimits &lim)
{
    use_time = false;
    soft_ms = hard_ms = 0;
    node_limit        = lim.nodes;
    if (lim.movetime > 0)
    {
        use_time = true;
        hard_ms = soft_ms = std::max<int64_t>(1, lim.movetime - move_overhead);
    }
    else if (lim.time[root.stm] > 0)
    {
        use_time          = true;
        int64_t remaining = lim.time[root.stm], increment = lim.inc[root.stm];
        int     moves_to_go = lim.movestogo > 0 ? lim.movestogo : 30;
        int64_t budget      = remaining / moves_to_go + increment * 3 / 4;
        soft_ms             = std::max<int64_t>(1, budget - move_overhead);
        hard_ms             = std::max<int64_t>(1, std::min<int64_t>(remaining - move_overhead, soft_ms * 4));
    }
}

bool Searcher::is_draw(const Position &pos) const
{
    if (pos.halfmove >= 100)
    {
        return true;
    }
    // Insufficient material (K vs K, K+minor vs K/K+minor).
    if (!(pos.by_type[PAWN] | pos.by_type[ROOK] | pos.by_type[QUEEN]))
    {
        int white_minors = popcount(pos.by_color[WHITE] & (pos.by_type[KNIGHT] | pos.by_type[BISHOP]));
        int black_minors = popcount(pos.by_color[BLACK] & (pos.by_type[KNIGHT] | pos.by_type[BISHOP]));
        if (white_minors <= 1 && black_minors <= 1)
        {
            return true;
        }
    }
    // Repetition: scan the path back to the last irreversible move (step 2 keeps the side to move).
    int end     = (int)hist.size();
    int stop_at = std::max(0, end - pos.halfmove);
    for (int i = end - 2; i >= stop_at; i -= 2)
    {
        if (hist[i] == pos.key)
        {
            return true;
        }
    }
    return false;
}

void Searcher::update_pv(int ply, Move move)
{
    pv_table[ply][0] = move;
    std::memcpy(&pv_table[ply][1], &pv_table[ply + 1][0], pv_len[ply + 1] * sizeof(Move));
    pv_len[ply] = pv_len[ply + 1] + 1;
}

int Searcher::qsearch(Position &pos, int alpha, int beta, int ply)
{
    if (time_up())
    {
        return 0; // time_up() already set g_stop for the main thread
    }
    nodes++;
    if (ply > seldepth)
    {
        seldepth = ply;
    }
    if (ply >= MAX_PLY - 1)
    {
        return evaluate(pos);
    }

    Bitboard checkers = pos.attackers_to(pos.king_sq(pos.stm), ~pos.stm, pos.occupied());
    Bitboard pinned   = pos.pinned_to_king();
    bool     in_check = checkers != 0;
    int      best     = -VALUE_INF;
    if (!in_check)
    {
        best = evaluate(pos);
        if (best >= beta)
        {
            return best;
        }
        if (best > alpha)
        {
            alpha = best;
        }
    }

    MoveList moves;
    generate_pseudo(pos, moves, !in_check); // in check: all evasions; else captures + promotions

    // MVV-LVA ordering.
    int scores[256];
    for (int index = 0; index < moves.size(); index++)
    {
        Move move  = moves[index];
        int  score = 0;
        if (move.is_capture())
        {
            score = 100 * SeeValue[move.is_ep() ? PAWN : type_of(pos.board[move.to()])] -
                    SeeValue[type_of(pos.board[move.from()])];
        }
        if (move.is_promo())
        {
            score += 900000 + move.promo_pt();
        }
        scores[index] = score;
    }

    int legal_count = 0;
    for (int index = 0; index < moves.size(); index++)
    {
        int best_index = index;
        for (int other = index + 1; other < moves.size(); other++)
        {
            if (scores[other] > scores[best_index])
            {
                best_index = other;
            }
        }
        std::swap(moves[index], moves[best_index]);
        std::swap(scores[index], scores[best_index]);
        Move move = moves[index];

        // Copy-free legality first, then SEE pruning, so only searched moves pay make_move.
        if (!pos.is_legal_fast(move, checkers, pinned))
        {
            continue;
        }
        legal_count++;

        if (!in_check && move.is_capture() && static_exchange_eval(pos, move) < 0)
        {
            continue; // skip losing captures
        }

        Position child = pos;
        child.make_move(move);
        int score = -qsearch(child, -beta, -alpha, ply + 1);
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
    if (in_check && legal_count == 0)
    {
        return -VALUE_MATE + ply; // checkmate (all evasions were illegal)
    }
    return best;
}

int Searcher::negamax(Position &pos, int depth, int alpha, int beta, int ply, bool cutnode, Move prev_move,
                      Move excluded)
{
    if (time_up())
    {
        return 0; // time_up() already set g_stop for the main thread
    }
    bool root    = ply == 0;
    bool pv_node = beta - alpha > 1;
    pv_len[ply]  = 0;

    if (!root)
    {
        if (is_draw(pos))
        {
            return draw_value();
        }
        // Mate-distance pruning.
        alpha = std::max(alpha, -VALUE_MATE + ply);
        beta  = std::min(beta, VALUE_MATE - ply - 1);
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
        return qsearch(pos, alpha, beta, ply);
    }

    nodes++;
    // Checkers + pinned once per node so the move loop can test legality without a copy-make (is_legal_fast),
    // and prune before paying make_move.
    Bitboard checkers = pos.attackers_to(pos.king_sq(pos.stm), ~pos.stm, pos.occupied());
    Bitboard pinned   = pos.pinned_to_king();
    bool     in_check = checkers != 0;

    // Continuation-history / countermove key = the (piece, to-square) of the move that reached this node.
    int prev_piece_to = -1;
    if (!prev_move.is_none())
    {
        Piece prev_piece = pos.board[prev_move.to()];
        if (prev_piece != NO_PIECE)
        {
            prev_piece_to = prev_piece * 64 + prev_move.to();
        }
    }

    TTData tt_entry{};
    bool   tt_hit   = TT.probe(pos.key, tt_entry);
    int    tt_score = tt_hit ? score_from_tt(int(tt_entry.score), ply) : VALUE_NONE;
    Move   tt_move  = tt_hit ? Move(uint16_t(tt_entry.move)) : Move::none();
    if (excluded.is_none() && !pv_node && tt_hit && int(tt_entry.depth) >= depth &&
        (tt_entry.bound == BOUND_EXACT || (tt_entry.bound == BOUND_LOWER && tt_score >= beta) ||
         (tt_entry.bound == BOUND_UPPER && tt_score <= alpha)))
    {
        return tt_score;
    }

    // Internal iterative reduction: with no TT move to anchor ordering at higher depths, search shallower
    // first so the cheaper search populates the TT move for the re-search.
    if (depth >= 4 && tt_move.is_none() && !in_check)
    {
        depth--;
    }

    // Raw static eval (stored in the TT); the corrected eval drives pruning/reductions. Keep them separate
    // so re-reading the TT eval never double-applies the correction.
    int raw_eval = in_check ? VALUE_NONE : (tt_hit && tt_entry.eval != VALUE_NONE ? tt_entry.eval : evaluate(pos));
    int eval     = raw_eval;
    if (!in_check)
    {
        eval += correction_history[pos.stm][pos.pawn_key & (CORRHIST_SIZE - 1)] / CORRHIST_GRAIN;
        eval = std::clamp(eval, -VALUE_MATE_IN_MAX + 1, VALUE_MATE_IN_MAX - 1);
    }

    // Reverse futility pruning (static null move).
    if (!pv_node && !in_check && depth <= 8 && !is_mate_score(beta) && eval - g_params.rfp_margin * depth >= beta)
    {
        return eval;
    }

    // Null-move pruning.
    if (!pv_node && !in_check && depth >= 3 && eval >= beta && pos.has_non_pawn_material(pos.stm))
    {
        int      reduction  = 3 + depth / 3 + std::min((eval - beta) / g_params.nmp_divisor, 3);
        Position null_child = pos;
        null_child.make_null();
        hist.push_back(pos.key);
        int score = -negamax(null_child, depth - reduction, -beta, -beta + 1, ply + 1, !cutnode, Move::none());
        hist.pop_back();
        if (g_stop)
        {
            return 0;
        }
        if (score >= beta)
        {
            return is_mate_score(score) ? beta : score;
        }
    }

    MoveList moves;
    generate_pseudo(pos, moves); // legality is filtered in the loop via the single make_move

    // Score moves: TT move, captures (MVV-LVA), killers, history.
    int scores[256];
    for (int index = 0; index < moves.size(); index++)
    {
        Move move = moves[index];
        int  move_score;
        if (move == tt_move)
        {
            move_score = 2000000000;
        }
        else if (move.is_capture())
        {
            move_score = 1000000 + 100 * SeeValue[move.is_ep() ? PAWN : type_of(pos.board[move.to()])] -
                         SeeValue[type_of(pos.board[move.from()])];
        }
        else if (move.is_promo())
        {
            move_score = 900000 + move.promo_pt();
        }
        else if (move == killers[ply][0])
        {
            move_score = 800000;
        }
        else if (move == killers[ply][1])
        {
            move_score = 700000;
        }
        else if (prev_piece_to >= 0 && move == counter_moves[prev_piece_to])
        {
            move_score = 650000;
        }
        else
        {
            int current_piece_to = pos.board[move.from()] * 64 + move.to();
            move_score           = history[pos.stm][move.from()][move.to()] +
                         (prev_piece_to >= 0 ? cont_hist[prev_piece_to * 768 + current_piece_to] : 0);
        }
        scores[index] = move_score;
    }

    int  best_score = -VALUE_INF;
    Move best_move  = Move::none();
    int  orig_alpha = alpha;
    int  move_count = 0;
    Move quiets[64];
    int  quiet_count = 0;

    for (int index = 0; index < moves.size(); index++)
    {
        int best_index = index;
        for (int other = index + 1; other < moves.size(); other++)
        {
            if (scores[other] > scores[best_index])
            {
                best_index = other;
            }
        }
        std::swap(moves[index], moves[best_index]);
        std::swap(scores[index], scores[best_index]);
        Move move = moves[index];
        if (move == excluded)
        {
            continue; // singular exclusion search: skip the move being tested for singularity
        }

        // Copy-free legality: skip illegal pseudo-legal moves BEFORE any make_move, so the LMP/futility/SEE
        // pruning below runs first and only searched moves pay the ~2KB copy. Skipping before move_count++
        // keeps legal-move ordering/pruning identical to a legal generator.
        if (!pos.is_legal_fast(move, checkers, pinned))
        {
            continue;
        }
        bool quiet = move.is_quiet();
        move_count++;

        // Late-move pruning: at low depth, stop trying quiet moves once deep into the ordered list.
        if (!pv_node && !in_check && quiet && depth <= 8 && move_count > g_params.lmp_base + depth * depth &&
            !is_mate_score(best_score))
        {
            continue;
        }

        // Futility pruning: at low depth, skip quiet moves that a margin cannot lift to alpha.
        if (!root && !pv_node && !in_check && quiet && depth <= 6 && move_count > 1 && !is_mate_score(best_score) &&
            eval + g_params.futility_base + g_params.futility_margin * depth <= alpha)
        {
            continue;
        }

        // SEE pruning of clearly-losing captures at low depth.
        if (!root && depth <= 6 && move.is_capture() && !is_mate_score(best_score) &&
            static_exchange_eval(pos, move) < -g_params.see_capture_margin * depth)
        {
            continue;
        }

        // Move survived pruning — make it now (legality already established above).
        Position child = pos;
        child.make_move(move);
        bool child_check = child.in_check();
        int  extension   = child_check ? 1 : 0;

        // Singular extension: if the TT move is much better than every alternative — an exclusion search
        // (this position without the TT move) at reduced depth fails low below a margin — extend it.
        if (!root && move == tt_move && excluded.is_none() && depth >= 8 && tt_hit &&
            int(tt_entry.depth) >= depth - 3 && (tt_entry.bound == BOUND_LOWER || tt_entry.bound == BOUND_EXACT) &&
            !is_mate_score(tt_score))
        {
            int singular_beta = tt_score - g_params.singular_margin * depth;
            int singular_score =
                negamax(pos, (depth - 1) / 2, singular_beta - 1, singular_beta, ply, cutnode, prev_move, tt_move);
            if (singular_score < singular_beta)
            {
                extension = 1;
            }
        }

        int new_depth = depth - 1 + extension;

        hist.push_back(pos.key);
        int score;
        if (move_count == 1)
        {
            score = -negamax(child, new_depth, -beta, -alpha, ply + 1, false, move);
        }
        else
        {
            int reduction = 0;
            if (depth >= 3 && move_count >= 4 && quiet && !in_check)
            {
                reduction = Reductions[std::min(depth, MAX_PLY - 1)][std::min(move_count, 63)];
                if (pv_node)
                {
                    reduction--;
                }
                if (cutnode)
                {
                    reduction++;
                }
                reduction = std::clamp(reduction, 0, new_depth - 1);
            }
            score = -negamax(child, new_depth - reduction, -alpha - 1, -alpha, ply + 1, true, move);
            if (score > alpha && reduction > 0)
            {
                score = -negamax(child, new_depth, -alpha - 1, -alpha, ply + 1, !cutnode, move);
            }
            if (score > alpha && score < beta)
            {
                score = -negamax(child, new_depth, -beta, -alpha, ply + 1, false, move);
            }
        }
        hist.pop_back();
        if (g_stop)
        {
            return 0;
        }

        if (quiet && quiet_count < 64)
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
                if (pv_node)
                {
                    update_pv(ply, move);
                }
                if (score >= beta)
                {
                    // Beta cutoff: reward the quiet cutoff move (main + continuation history, killer,
                    // countermove), punish the quiets that failed before it.
                    if (quiet)
                    {
                        if (killers[ply][0] != move)
                        {
                            killers[ply][1] = killers[ply][0];
                            killers[ply][0] = move;
                        }
                        if (prev_piece_to >= 0)
                        {
                            counter_moves[prev_piece_to] = move;
                        }
                        int  bonus         = std::min(depth * depth, g_params.history_max);
                        auto apply_gravity = [&](int &entry, int change) {
                            entry += change - entry * std::abs(change) / 16384;
                        };
                        int current_piece_to = pos.board[move.from()] * 64 + move.to();
                        apply_gravity(history[pos.stm][move.from()][move.to()], bonus);
                        if (prev_piece_to >= 0)
                        {
                            apply_gravity(cont_hist[prev_piece_to * 768 + current_piece_to], bonus);
                        }
                        for (int quiet_index = 0; quiet_index < quiet_count - 1; quiet_index++)
                        {
                            Move quiet_move = quiets[quiet_index];
                            apply_gravity(history[pos.stm][quiet_move.from()][quiet_move.to()], -bonus);
                            if (prev_piece_to >= 0)
                            {
                                apply_gravity(cont_hist[prev_piece_to * 768 + pos.board[quiet_move.from()] * 64 +
                                                        quiet_move.to()],
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
        return in_check ? -VALUE_MATE + ply : draw_value(); // no legal move: checkmate or stalemate
    }

    Bound bound = best_score >= beta ? BOUND_LOWER : (alpha > orig_alpha ? BOUND_EXACT : BOUND_UPPER);
    if (excluded.is_none())
    {
        TT.store(pos.key, best_score, in_check ? VALUE_NONE : raw_eval, depth, bound, best_move, ply);

        // Update the eval correction: blend in (search score - raw static eval), but only when the score is
        // a trustworthy signal — not in check, not a tactical (capture) best move, not a mate, and the bound
        // does not contradict the direction of the correction.
        if (!in_check && !is_mate_score(best_score) && (best_move.is_none() || !best_move.is_capture()) &&
            !(bound == BOUND_LOWER && best_score <= raw_eval) && !(bound == BOUND_UPPER && best_score >= raw_eval))
        {
            int &entry  = correction_history[pos.stm][pos.pawn_key & (CORRHIST_SIZE - 1)];
            int  target = std::clamp((best_score - raw_eval) * CORRHIST_GRAIN, -CORRHIST_MAX, CORRHIST_MAX);
            int  weight = std::min(depth + 1, 16);
            entry       = std::clamp((entry * (256 - weight) + target * weight) / 256, -CORRHIST_MAX, CORRHIST_MAX);
        }
    }
    if (root)
    {
        root_best = best_move;
    }
    return best_score;
}

Move Searcher::go(Position root, const SearchLimits &lim, bool is_main_thread)
{
    is_main = is_main_thread;
    if (is_main)
    {
        g_stop = false; // clear the shared stop before a new search (helpers are launched after this)
    }
    nodes    = 0;
    seldepth = 0;
    std::memset(killers, 0, sizeof(killers));
    std::memset(history, 0, sizeof(history));
    std::memset(correction_history, 0, sizeof(correction_history));
    std::memset(counter_moves, 0, sizeof(counter_moves));
    std::fill(cont_hist.begin(), cont_hist.end(), 0);
    start = std::chrono::steady_clock::now();
    set_time(root, lim);
    if (is_main)
    {
        TT.new_search(); // bump generation once per search, not per helper thread
    }
    if (nnue::is_loaded())
    {
        nnue::refresh(root.acc, root); // authoritative root accumulator (robust to a net loaded mid-game)
    }

    root_best      = Move::none();
    Move best      = Move::none();
    int  max_depth = lim.depth > 0 ? lim.depth : MAX_PLY - 2;
    int  score     = 0;

    for (int depth = 1; depth <= max_depth; depth++)
    {
        // Aspiration windows once we have a score to trust.
        int alpha = -VALUE_INF, beta = VALUE_INF, delta = g_params.aspiration_delta;
        if (depth >= 4)
        {
            alpha = std::max(-VALUE_INF, score - delta);
            beta  = std::min(VALUE_INF, score + delta);
        }
        while (true)
        {
            int window_score = negamax(root, depth, alpha, beta, 0, false, Move::none());
            if (g_stop)
            {
                break;
            }
            score = window_score;
            if (window_score <= alpha)
            {
                beta  = (alpha + beta) / 2;
                alpha = std::max(-VALUE_INF, window_score - delta);
                delta += delta / 2;
            }
            else if (window_score >= beta)
            {
                beta = std::min(VALUE_INF, window_score + delta);
                delta += delta / 2;
            }
            else
            {
                break;
            }
        }
        if (g_stop && best != Move::none())
        {
            break;
        }
        best = root_best;

        if (is_main && !silent)
        {
            int64_t  elapsed_ms       = elapsed();
            uint64_t nodes_per_second = elapsed_ms ? nodes * 1000 / elapsed_ms : nodes;
            // Score string.
            char score_str[32];
            if (is_mate_score(score))
            {
                int mate = score > 0 ? (VALUE_MATE - score + 1) / 2 : -(VALUE_MATE + score) / 2;
                snprintf(score_str, sizeof score_str, "mate %d", mate);
            }
            else
            {
                snprintf(score_str, sizeof score_str, "cp %d", score);
            }
            printf("info depth %d seldepth %d score %s nodes %llu nps %llu time %lld hashfull %d pv", depth, seldepth,
                   score_str, (unsigned long long)nodes, (unsigned long long)nodes_per_second, (long long)elapsed_ms,
                   TT.hashfull());
            for (int pv_index = 0; pv_index < pv_len[0]; pv_index++)
            {
                printf(" %s", pv_table[0][pv_index].to_uci().c_str());
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
        if (is_main && use_time && elapsed() >= soft_ms)
        {
            break; // don't start a deeper iteration we can't finish
        }
        if (is_main && is_mate_score(score) && lim.depth == 0 && !lim.infinite)
        {
            break;
        }
    }
    if (is_main)
    {
        g_stop = true; // release the helper threads
    }
    root_score = score;
    return best.is_none() ? root_best : best;
}
