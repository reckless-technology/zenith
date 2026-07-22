#include "search.h"
#include "bitboard.h"
#include "eval.h"
#include "nnue.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

std::atomic<bool> g_stop{false};

namespace
{

int       Reductions[MAX_PLY][64];
const int SeeValue[6] = {100, 320, 330, 500, 900, 10000};

// Eval correction history: a running average of (search score - static eval) keyed by pawn structure, used
// to nudge the static eval toward what search has historically found. Entries are cp * CORRHIST_GRAIN.
constexpr int CORRHIST_SIZE  = 16384; // power of two -> index by (pawnKey & (SIZE-1))
constexpr int CORRHIST_GRAIN = 256;
constexpr int CORRHIST_MAX   = 64 * CORRHIST_GRAIN; // clamp the correction to +/-64 cp

int draw_value()
{
    return 0;
}

// Static Exchange Evaluation of a capture: net material after the optimal capture sequence on m.to().
int see(const Position &pos, Move m)
{
    int to = m.to(), from = m.from();
    int captured = m.is_ep() ? SeeValue[PAWN] : SeeValue[type_of(pos.board[to])];
    int gain[32];
    int d   = 0;
    gain[0] = captured;

    Bitboard occ = pos.occupied();
    if (m.is_ep())
    {
        occ ^= sq_bb(to + (pos.stm == WHITE ? -8 : 8));
    }
    occ ^= sq_bb(from);

    PieceType attacker = type_of(pos.board[from]);
    Color     side     = ~pos.stm;
    // Attackers hitting `to` given the current occupancy; recomputed each ply so x-rays reveal.
    Bitboard attackers = (pos.attackers_to(to, WHITE, occ) | pos.attackers_to(to, BLACK, occ)) & occ;

    while (true)
    {
        d++;
        gain[d]          = SeeValue[attacker] - gain[d - 1];
        Bitboard sideAtt = attackers & pos.byColor[side];
        if (!sideAtt)
        {
            break;
        }
        // least valuable attacker of `side`
        PieceType lva = KING;
        Bitboard  one = 0;
        for (int pt = PAWN; pt <= KING; pt++)
        {
            Bitboard b = sideAtt & pos.pieces(side, PieceType(pt));
            if (b)
            {
                lva = PieceType(pt);
                one = b & (~b + 1);
                break;
            }
        }
        attacker = lva;
        occ ^= one;
        attackers = (pos.attackers_to(to, WHITE, occ) | pos.attackers_to(to, BLACK, occ)) & occ;
        side      = ~side;
        if (attacker == KING && (attackers & pos.byColor[side]))
        {
            // Cannot recapture with the king into a still-defended square; stop before it.
            break;
        }
    }
    while (--d > 0)
    {
        gain[d - 1] = -std::max(-gain[d - 1], gain[d]);
    }
    return gain[0];
}

} // namespace

void init_search()
{
    for (int d = 1; d < MAX_PLY; d++)
    {
        for (int m = 1; m < 64; m++)
        {
            Reductions[d][m] = int(0.80 + std::log(d) * std::log(m) / 2.30);
        }
    }
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
    if (isMain && ((nodeLimit && nodes >= (uint64_t)nodeLimit) || (useTime && elapsed() >= hardMs)))
    {
        g_stop.store(true, std::memory_order_relaxed);
        return true;
    }
    return false;
}

void Searcher::set_time(const Position &root, const SearchLimits &lim)
{
    useTime = false;
    softMs = hardMs = 0;
    nodeLimit       = lim.nodes;
    if (lim.movetime > 0)
    {
        useTime = true;
        hardMs = softMs = std::max<int64_t>(1, lim.movetime - moveOverhead);
    }
    else if (lim.time[root.stm] > 0)
    {
        useTime   = true;
        int64_t t = lim.time[root.stm], inc = lim.inc[root.stm];
        int     mtg    = lim.movestogo > 0 ? lim.movestogo : 30;
        int64_t budget = t / mtg + inc * 3 / 4;
        softMs         = std::max<int64_t>(1, budget - moveOverhead);
        hardMs         = std::max<int64_t>(1, std::min<int64_t>(t - moveOverhead, softMs * 4));
    }
}

bool Searcher::is_draw(const Position &pos) const
{
    if (pos.halfmove >= 100)
    {
        return true;
    }
    // Insufficient material (K vs K, K+minor vs K/K+minor).
    if (!(pos.byType[PAWN] | pos.byType[ROOK] | pos.byType[QUEEN]))
    {
        int wm = popcount(pos.byColor[WHITE] & (pos.byType[KNIGHT] | pos.byType[BISHOP]));
        int bm = popcount(pos.byColor[BLACK] & (pos.byType[KNIGHT] | pos.byType[BISHOP]));
        if (wm <= 1 && bm <= 1)
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

void Searcher::update_pv(int ply, Move m)
{
    pvTable[ply][0] = m;
    std::memcpy(&pvTable[ply][1], &pvTable[ply + 1][0], pvLen[ply + 1] * sizeof(Move));
    pvLen[ply] = pvLen[ply + 1] + 1;
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
    bool     inCheck  = checkers != 0;
    int      best     = -VALUE_INF;
    if (!inCheck)
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
    generate_pseudo(pos, moves, !inCheck); // in check: all evasions; else captures + promotions

    // MVV-LVA ordering.
    int scores[256];
    for (int i = 0; i < moves.size(); i++)
    {
        Move m = moves[i];
        int  s = 0;
        if (m.is_capture())
        {
            s = 100 * SeeValue[m.is_ep() ? PAWN : type_of(pos.board[m.to()])] - SeeValue[type_of(pos.board[m.from()])];
        }
        if (m.is_promo())
        {
            s += 900000 + m.promo_pt();
        }
        scores[i] = s;
    }

    int legalCount = 0;
    for (int i = 0; i < moves.size(); i++)
    {
        int bi = i;
        for (int j = i + 1; j < moves.size(); j++)
        {
            if (scores[j] > scores[bi])
            {
                bi = j;
            }
        }
        std::swap(moves[i], moves[bi]);
        std::swap(scores[i], scores[bi]);
        Move m = moves[i];

        // Copy-free legality first, then SEE pruning, so only searched moves pay make_move.
        if (!pos.is_legal_fast(m, checkers, pinned))
        {
            continue;
        }
        legalCount++;

        if (!inCheck && m.is_capture() && see(pos, m) < 0)
        {
            continue; // skip losing captures
        }

        Position child = pos;
        child.make_move(m);
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
    if (inCheck && legalCount == 0)
    {
        return -VALUE_MATE + ply; // checkmate (all evasions were illegal)
    }
    return best;
}

int Searcher::negamax(Position &pos, int depth, int alpha, int beta, int ply, bool cutnode, Move prevMove,
                      Move excluded)
{
    if (time_up())
    {
        return 0; // time_up() already set g_stop for the main thread
    }
    bool root   = ply == 0;
    bool pvNode = beta - alpha > 1;
    pvLen[ply]  = 0;

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
    bool     inCheck  = checkers != 0;

    // Continuation-history / countermove key = the (piece, to-square) of the move that reached this node.
    int prevPT = -1;
    if (!prevMove.is_none())
    {
        Piece pp = pos.board[prevMove.to()];
        if (pp != NO_PIECE)
        {
            prevPT = pp * 64 + prevMove.to();
        }
    }

    TTEntry tte;
    bool    ttHit   = TT.probe(pos.key, tte);
    int     ttScore = ttHit ? score_from_tt(tte.score, ply) : VALUE_NONE;
    Move    ttMove  = ttHit ? Move(tte.move) : Move::none();
    if (excluded.is_none() && !pvNode && ttHit && tte.depth >= depth &&
        (tte.bound == BOUND_EXACT || (tte.bound == BOUND_LOWER && ttScore >= beta) ||
         (tte.bound == BOUND_UPPER && ttScore <= alpha)))
    {
        return ttScore;
    }

    // Internal iterative reduction: with no TT move to anchor ordering at higher depths, search shallower
    // first so the cheaper search populates the TT move for the re-search.
    if (depth >= 4 && ttMove.is_none() && !inCheck)
    {
        depth--;
    }

    // Raw static eval (stored in the TT); the corrected eval drives pruning/reductions. Keep them separate
    // so re-reading the TT eval never double-applies the correction.
    int rawEval = inCheck ? VALUE_NONE : (ttHit && tte.eval != VALUE_NONE ? tte.eval : evaluate(pos));
    int eval    = rawEval;
    if (!inCheck)
    {
        eval += correctionHistory[pos.stm][pos.pawnKey & (CORRHIST_SIZE - 1)] / CORRHIST_GRAIN;
        eval = std::clamp(eval, -VALUE_MATE_IN_MAX + 1, VALUE_MATE_IN_MAX - 1);
    }

    // Reverse futility pruning (static null move).
    if (!pvNode && !inCheck && depth <= 8 && !is_mate_score(beta) && eval - 80 * depth >= beta)
    {
        return eval;
    }

    // Null-move pruning.
    if (!pvNode && !inCheck && depth >= 3 && eval >= beta && pos.has_non_pawn_material(pos.stm))
    {
        int      R  = 3 + depth / 3 + std::min((eval - beta) / 200, 3);
        Position np = pos;
        np.make_null();
        hist.push_back(pos.key);
        int score = -negamax(np, depth - R, -beta, -beta + 1, ply + 1, !cutnode, Move::none());
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
    for (int i = 0; i < moves.size(); i++)
    {
        Move m = moves[i];
        int  s;
        if (m == ttMove)
        {
            s = 2000000000;
        }
        else if (m.is_capture())
        {
            s = 1000000 + 100 * SeeValue[m.is_ep() ? PAWN : type_of(pos.board[m.to()])] -
                SeeValue[type_of(pos.board[m.from()])];
        }
        else if (m.is_promo())
        {
            s = 900000 + m.promo_pt();
        }
        else if (m == killers[ply][0])
        {
            s = 800000;
        }
        else if (m == killers[ply][1])
        {
            s = 700000;
        }
        else if (prevPT >= 0 && m == counterMoves[prevPT])
        {
            s = 650000;
        }
        else
        {
            int curPT = pos.board[m.from()] * 64 + m.to();
            s         = history[pos.stm][m.from()][m.to()] + (prevPT >= 0 ? contHist[prevPT * 768 + curPT] : 0);
        }
        scores[i] = s;
    }

    int  bestScore = -VALUE_INF;
    Move bestMove  = Move::none();
    int  origAlpha = alpha;
    int  moveCount = 0;
    Move quiets[64];
    int  nQuiets = 0;

    for (int i = 0; i < moves.size(); i++)
    {
        int bi = i;
        for (int j = i + 1; j < moves.size(); j++)
        {
            if (scores[j] > scores[bi])
            {
                bi = j;
            }
        }
        std::swap(moves[i], moves[bi]);
        std::swap(scores[i], scores[bi]);
        Move m = moves[i];
        if (m == excluded)
        {
            continue; // singular exclusion search: skip the move being tested for singularity
        }

        // Copy-free legality: skip illegal pseudo-legal moves BEFORE any make_move, so the LMP/futility/SEE
        // pruning below runs first and only searched moves pay the ~2KB copy. Skipping before moveCount++
        // keeps legal-move ordering/pruning identical to a legal generator.
        if (!pos.is_legal_fast(m, checkers, pinned))
        {
            continue;
        }
        bool quiet = m.is_quiet();
        moveCount++;

        // Late-move pruning: at low depth, stop trying quiet moves once deep into the ordered list.
        if (!pvNode && !inCheck && quiet && depth <= 8 && moveCount > 3 + depth * depth && !is_mate_score(bestScore))
        {
            continue;
        }

        // Futility pruning: at low depth, skip quiet moves that a margin cannot lift to alpha.
        if (!root && !pvNode && !inCheck && quiet && depth <= 6 && moveCount > 1 && !is_mate_score(bestScore) &&
            eval + 100 + 90 * depth <= alpha)
        {
            continue;
        }

        // SEE pruning of clearly-losing captures at low depth.
        if (!root && depth <= 6 && m.is_capture() && !is_mate_score(bestScore) && see(pos, m) < -100 * depth)
        {
            continue;
        }

        // Move survived pruning — make it now (legality already established above).
        Position child = pos;
        child.make_move(m);
        bool childCheck = child.in_check();
        int  ext        = childCheck ? 1 : 0;

        // Singular extension: if the TT move is much better than every alternative — an exclusion search
        // (this position without the TT move) at reduced depth fails low below a margin — extend it.
        if (!root && m == ttMove && excluded.is_none() && depth >= 8 && ttHit && tte.depth >= depth - 3 &&
            (tte.bound == BOUND_LOWER || tte.bound == BOUND_EXACT) && !is_mate_score(ttScore))
        {
            int singularBeta = ttScore - 3 * depth;
            int singularScore =
                negamax(pos, (depth - 1) / 2, singularBeta - 1, singularBeta, ply, cutnode, prevMove, ttMove);
            if (singularScore < singularBeta)
            {
                ext = 1;
            }
        }

        int newDepth = depth - 1 + ext;

        hist.push_back(pos.key);
        int score;
        if (moveCount == 1)
        {
            score = -negamax(child, newDepth, -beta, -alpha, ply + 1, false, m);
        }
        else
        {
            int R = 0;
            if (depth >= 3 && moveCount >= 4 && quiet && !inCheck)
            {
                R = Reductions[std::min(depth, MAX_PLY - 1)][std::min(moveCount, 63)];
                if (pvNode)
                {
                    R--;
                }
                if (cutnode)
                {
                    R++;
                }
                R = std::clamp(R, 0, newDepth - 1);
            }
            score = -negamax(child, newDepth - R, -alpha - 1, -alpha, ply + 1, true, m);
            if (score > alpha && R > 0)
            {
                score = -negamax(child, newDepth, -alpha - 1, -alpha, ply + 1, !cutnode, m);
            }
            if (score > alpha && score < beta)
            {
                score = -negamax(child, newDepth, -beta, -alpha, ply + 1, false, m);
            }
        }
        hist.pop_back();
        if (g_stop)
        {
            return 0;
        }

        if (quiet && nQuiets < 64)
        {
            quiets[nQuiets++] = m;
        }

        if (score > bestScore)
        {
            bestScore = score;
            bestMove  = m;
            if (score > alpha)
            {
                alpha = score;
                if (pvNode)
                {
                    update_pv(ply, m);
                }
                if (score >= beta)
                {
                    // Beta cutoff: reward the quiet cutoff move (main + continuation history, killer,
                    // countermove), punish the quiets that failed before it.
                    if (quiet)
                    {
                        if (killers[ply][0] != m)
                        {
                            killers[ply][1] = killers[ply][0];
                            killers[ply][0] = m;
                        }
                        if (prevPT >= 0)
                        {
                            counterMoves[prevPT] = m;
                        }
                        int  bonus = std::min(depth * depth, 400);
                        auto grav  = [&](int &e, int b) { e += b - e * std::abs(b) / 16384; };
                        int  curPT = pos.board[m.from()] * 64 + m.to();
                        grav(history[pos.stm][m.from()][m.to()], bonus);
                        if (prevPT >= 0)
                        {
                            grav(contHist[prevPT * 768 + curPT], bonus);
                        }
                        for (int q = 0; q < nQuiets - 1; q++)
                        {
                            Move qm = quiets[q];
                            grav(history[pos.stm][qm.from()][qm.to()], -bonus);
                            if (prevPT >= 0)
                            {
                                grav(contHist[prevPT * 768 + pos.board[qm.from()] * 64 + qm.to()], -bonus);
                            }
                        }
                    }
                    break;
                }
            }
        }
    }

    if (moveCount == 0)
    {
        return inCheck ? -VALUE_MATE + ply : draw_value(); // no legal move: checkmate or stalemate
    }

    Bound b = bestScore >= beta ? BOUND_LOWER : (alpha > origAlpha ? BOUND_EXACT : BOUND_UPPER);
    if (excluded.is_none())
    {
        TT.store(pos.key, bestScore, inCheck ? VALUE_NONE : rawEval, depth, b, bestMove, ply);

        // Update the eval correction: blend in (search score - raw static eval), but only when the score is
        // a trustworthy signal — not in check, not a tactical (capture) best move, not a mate, and the bound
        // does not contradict the direction of the correction.
        if (!inCheck && !is_mate_score(bestScore) && (bestMove.is_none() || !bestMove.is_capture()) &&
            !(b == BOUND_LOWER && bestScore <= rawEval) && !(b == BOUND_UPPER && bestScore >= rawEval))
        {
            int &entry  = correctionHistory[pos.stm][pos.pawnKey & (CORRHIST_SIZE - 1)];
            int  target = std::clamp((bestScore - rawEval) * CORRHIST_GRAIN, -CORRHIST_MAX, CORRHIST_MAX);
            int  weight = std::min(depth + 1, 16);
            entry       = std::clamp((entry * (256 - weight) + target * weight) / 256, -CORRHIST_MAX, CORRHIST_MAX);
        }
    }
    if (root)
    {
        rootBest = bestMove;
    }
    return bestScore;
}

Move Searcher::go(Position root, const SearchLimits &lim, bool isMainThread)
{
    isMain = isMainThread;
    if (isMain)
    {
        g_stop = false; // clear the shared stop before a new search (helpers are launched after this)
    }
    nodes    = 0;
    seldepth = 0;
    std::memset(killers, 0, sizeof(killers));
    std::memset(history, 0, sizeof(history));
    std::memset(correctionHistory, 0, sizeof(correctionHistory));
    std::memset(counterMoves, 0, sizeof(counterMoves));
    std::fill(contHist.begin(), contHist.end(), 0);
    start = std::chrono::steady_clock::now();
    set_time(root, lim);
    if (isMain)
    {
        TT.new_search(); // bump generation once per search, not per helper thread
    }
    if (nnue::is_loaded())
    {
        nnue::refresh(root.acc, root); // authoritative root accumulator (robust to a net loaded mid-game)
    }

    rootBest      = Move::none();
    Move best     = Move::none();
    int  maxDepth = lim.depth > 0 ? lim.depth : MAX_PLY - 2;
    int  score    = 0;

    for (int depth = 1; depth <= maxDepth; depth++)
    {
        // Aspiration windows once we have a score to trust.
        int alpha = -VALUE_INF, beta = VALUE_INF, delta = 20;
        if (depth >= 4)
        {
            alpha = std::max(-VALUE_INF, score - delta);
            beta  = std::min(VALUE_INF, score + delta);
        }
        while (true)
        {
            int s = negamax(root, depth, alpha, beta, 0, false, Move::none());
            if (g_stop)
            {
                break;
            }
            score = s;
            if (s <= alpha)
            {
                beta  = (alpha + beta) / 2;
                alpha = std::max(-VALUE_INF, s - delta);
                delta += delta / 2;
            }
            else if (s >= beta)
            {
                beta = std::min(VALUE_INF, s + delta);
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
        best = rootBest;

        if (isMain && !silent)
        {
            int64_t  ms  = elapsed();
            uint64_t nps = ms ? nodes * 1000 / ms : nodes;
            // Score string.
            char scoreStr[32];
            if (is_mate_score(score))
            {
                int mate = score > 0 ? (VALUE_MATE - score + 1) / 2 : -(VALUE_MATE + score) / 2;
                snprintf(scoreStr, sizeof scoreStr, "mate %d", mate);
            }
            else
            {
                snprintf(scoreStr, sizeof scoreStr, "cp %d", score);
            }
            printf("info depth %d seldepth %d score %s nodes %llu nps %llu time %lld hashfull %d pv", depth, seldepth,
                   scoreStr, (unsigned long long)nodes, (unsigned long long)nps, (long long)ms, TT.hashfull());
            for (int i = 0; i < pvLen[0]; i++)
            {
                printf(" %s", pvTable[0][i].to_uci().c_str());
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
        if (isMain && useTime && elapsed() >= softMs)
        {
            break; // don't start a deeper iteration we can't finish
        }
        if (isMain && is_mate_score(score) && lim.depth == 0 && !lim.infinite)
        {
            break;
        }
    }
    if (isMain)
    {
        g_stop = true; // release the helper threads
    }
    rootScore = score;
    return best.is_none() ? rootBest : best;
}
