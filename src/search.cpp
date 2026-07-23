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
constexpr int CORRHIST_SIZE  = 16384; // power of two -> index by (pawnKey & (SIZE-1))
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
    int swapIndex = 0;
    gain[0]       = captured;

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
        swapIndex++;
        gain[swapIndex]        = SeeValue[attacker] - gain[swapIndex - 1];
        Bitboard sideAttackers = attackers & pos.byColor[side];
        if (!sideAttackers)
        {
            break;
        }
        // Least valuable attacker of `side`.
        PieceType leastValuableAttacker = KING;
        Bitboard  attackerBit           = 0;
        for (int pieceType = PAWN; pieceType <= KING; pieceType++)
        {
            Bitboard candidates = sideAttackers & pos.pieces(side, PieceType(pieceType));
            if (candidates)
            {
                leastValuableAttacker = PieceType(pieceType);
                attackerBit           = candidates & (~candidates + 1);
                break;
            }
        }
        attacker = leastValuableAttacker;
        occupied ^= attackerBit;
        attackers = (pos.attackers_to(to, WHITE, occupied) | pos.attackers_to(to, BLACK, occupied)) & occupied;
        side      = ~side;
        if (attacker == KING && (attackers & pos.byColor[side]))
        {
            // Cannot recapture with the king into a still-defended square; stop before it.
            break;
        }
    }
    while (--swapIndex > 0)
    {
        gain[swapIndex - 1] = -std::max(-gain[swapIndex - 1], gain[swapIndex]);
    }
    return gain[0];
}

} // namespace

void init_search()
{
    double lmrBase    = g_params.lmrBaseX100 / 100.0;
    double lmrDivisor = g_params.lmrDivisorX100 / 100.0;
    for (int depth = 1; depth < MAX_PLY; depth++)
    {
        for (int moveNumber = 1; moveNumber < 64; moveNumber++)
        {
            Reductions[depth][moveNumber] = int(lmrBase + std::log(depth) * std::log(moveNumber) / lmrDivisor);
        }
    }
}

bool set_search_param(const std::string &name, int value)
{
    if (name == "RfpMargin")
    {
        g_params.rfpMargin = value;
    }
    else if (name == "NmpDivisor")
    {
        g_params.nmpDivisor = value;
    }
    else if (name == "LmpBase")
    {
        g_params.lmpBase = value;
    }
    else if (name == "FutilityBase")
    {
        g_params.futilityBase = value;
    }
    else if (name == "FutilityMargin")
    {
        g_params.futilityMargin = value;
    }
    else if (name == "SeeCaptureMargin")
    {
        g_params.seeCaptureMargin = value;
    }
    else if (name == "SingularMargin")
    {
        g_params.singularMargin = value;
    }
    else if (name == "AspirationDelta")
    {
        g_params.aspirationDelta = value;
    }
    else if (name == "HistoryMax")
    {
        g_params.historyMax = value;
    }
    else if (name == "LmrBase")
    {
        g_params.lmrBaseX100 = value;
        init_search(); // LMR table depends on this
    }
    else if (name == "LmrDivisor")
    {
        g_params.lmrDivisorX100 = value;
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
        useTime           = true;
        int64_t remaining = lim.time[root.stm], increment = lim.inc[root.stm];
        int     movesToGo = lim.movestogo > 0 ? lim.movestogo : 30;
        int64_t budget    = remaining / movesToGo + increment * 3 / 4;
        softMs            = std::max<int64_t>(1, budget - moveOverhead);
        hardMs            = std::max<int64_t>(1, std::min<int64_t>(remaining - moveOverhead, softMs * 4));
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
        int whiteMinors = popcount(pos.byColor[WHITE] & (pos.byType[KNIGHT] | pos.byType[BISHOP]));
        int blackMinors = popcount(pos.byColor[BLACK] & (pos.byType[KNIGHT] | pos.byType[BISHOP]));
        if (whiteMinors <= 1 && blackMinors <= 1)
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
    pvTable[ply][0] = move;
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

    int legalCount = 0;
    for (int index = 0; index < moves.size(); index++)
    {
        int bestIndex = index;
        for (int other = index + 1; other < moves.size(); other++)
        {
            if (scores[other] > scores[bestIndex])
            {
                bestIndex = other;
            }
        }
        std::swap(moves[index], moves[bestIndex]);
        std::swap(scores[index], scores[bestIndex]);
        Move move = moves[index];

        // Copy-free legality first, then SEE pruning, so only searched moves pay make_move.
        if (!pos.is_legal_fast(move, checkers, pinned))
        {
            continue;
        }
        legalCount++;

        if (!inCheck && move.is_capture() && static_exchange_eval(pos, move) < 0)
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
    int prevPieceTo = -1;
    if (!prevMove.is_none())
    {
        Piece prevPiece = pos.board[prevMove.to()];
        if (prevPiece != NO_PIECE)
        {
            prevPieceTo = prevPiece * 64 + prevMove.to();
        }
    }

    TTEntry ttEntry;
    bool    ttHit   = TT.probe(pos.key, ttEntry);
    int     ttScore = ttHit ? score_from_tt(ttEntry.score, ply) : VALUE_NONE;
    Move    ttMove  = ttHit ? Move(ttEntry.move) : Move::none();
    if (excluded.is_none() && !pvNode && ttHit && ttEntry.depth >= depth &&
        (ttEntry.bound == BOUND_EXACT || (ttEntry.bound == BOUND_LOWER && ttScore >= beta) ||
         (ttEntry.bound == BOUND_UPPER && ttScore <= alpha)))
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
    int rawEval = inCheck ? VALUE_NONE : (ttHit && ttEntry.eval != VALUE_NONE ? ttEntry.eval : evaluate(pos));
    int eval    = rawEval;
    if (!inCheck)
    {
        eval += correctionHistory[pos.stm][pos.pawnKey & (CORRHIST_SIZE - 1)] / CORRHIST_GRAIN;
        eval = std::clamp(eval, -VALUE_MATE_IN_MAX + 1, VALUE_MATE_IN_MAX - 1);
    }

    // Reverse futility pruning (static null move).
    if (!pvNode && !inCheck && depth <= 8 && !is_mate_score(beta) && eval - g_params.rfpMargin * depth >= beta)
    {
        return eval;
    }

    // Null-move pruning.
    if (!pvNode && !inCheck && depth >= 3 && eval >= beta && pos.has_non_pawn_material(pos.stm))
    {
        int      reduction = 3 + depth / 3 + std::min((eval - beta) / g_params.nmpDivisor, 3);
        Position nullChild = pos;
        nullChild.make_null();
        hist.push_back(pos.key);
        int score = -negamax(nullChild, depth - reduction, -beta, -beta + 1, ply + 1, !cutnode, Move::none());
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
        int  moveScore;
        if (move == ttMove)
        {
            moveScore = 2000000000;
        }
        else if (move.is_capture())
        {
            moveScore = 1000000 + 100 * SeeValue[move.is_ep() ? PAWN : type_of(pos.board[move.to()])] -
                        SeeValue[type_of(pos.board[move.from()])];
        }
        else if (move.is_promo())
        {
            moveScore = 900000 + move.promo_pt();
        }
        else if (move == killers[ply][0])
        {
            moveScore = 800000;
        }
        else if (move == killers[ply][1])
        {
            moveScore = 700000;
        }
        else if (prevPieceTo >= 0 && move == counterMoves[prevPieceTo])
        {
            moveScore = 650000;
        }
        else
        {
            int currentPieceTo = pos.board[move.from()] * 64 + move.to();
            moveScore          = history[pos.stm][move.from()][move.to()] +
                        (prevPieceTo >= 0 ? contHist[prevPieceTo * 768 + currentPieceTo] : 0);
        }
        scores[index] = moveScore;
    }

    int  bestScore = -VALUE_INF;
    Move bestMove  = Move::none();
    int  origAlpha = alpha;
    int  moveCount = 0;
    Move quiets[64];
    int  quietCount = 0;

    for (int index = 0; index < moves.size(); index++)
    {
        int bestIndex = index;
        for (int other = index + 1; other < moves.size(); other++)
        {
            if (scores[other] > scores[bestIndex])
            {
                bestIndex = other;
            }
        }
        std::swap(moves[index], moves[bestIndex]);
        std::swap(scores[index], scores[bestIndex]);
        Move move = moves[index];
        if (move == excluded)
        {
            continue; // singular exclusion search: skip the move being tested for singularity
        }

        // Copy-free legality: skip illegal pseudo-legal moves BEFORE any make_move, so the LMP/futility/SEE
        // pruning below runs first and only searched moves pay the ~2KB copy. Skipping before moveCount++
        // keeps legal-move ordering/pruning identical to a legal generator.
        if (!pos.is_legal_fast(move, checkers, pinned))
        {
            continue;
        }
        bool quiet = move.is_quiet();
        moveCount++;

        // Late-move pruning: at low depth, stop trying quiet moves once deep into the ordered list.
        if (!pvNode && !inCheck && quiet && depth <= 8 && moveCount > g_params.lmpBase + depth * depth &&
            !is_mate_score(bestScore))
        {
            continue;
        }

        // Futility pruning: at low depth, skip quiet moves that a margin cannot lift to alpha.
        if (!root && !pvNode && !inCheck && quiet && depth <= 6 && moveCount > 1 && !is_mate_score(bestScore) &&
            eval + g_params.futilityBase + g_params.futilityMargin * depth <= alpha)
        {
            continue;
        }

        // SEE pruning of clearly-losing captures at low depth.
        if (!root && depth <= 6 && move.is_capture() && !is_mate_score(bestScore) &&
            static_exchange_eval(pos, move) < -g_params.seeCaptureMargin * depth)
        {
            continue;
        }

        // Move survived pruning — make it now (legality already established above).
        Position child = pos;
        child.make_move(move);
        bool childCheck = child.in_check();
        int  extension  = childCheck ? 1 : 0;

        // Singular extension: if the TT move is much better than every alternative — an exclusion search
        // (this position without the TT move) at reduced depth fails low below a margin — extend it.
        if (!root && move == ttMove && excluded.is_none() && depth >= 8 && ttHit && ttEntry.depth >= depth - 3 &&
            (ttEntry.bound == BOUND_LOWER || ttEntry.bound == BOUND_EXACT) && !is_mate_score(ttScore))
        {
            int singularBeta = ttScore - g_params.singularMargin * depth;
            int singularScore =
                negamax(pos, (depth - 1) / 2, singularBeta - 1, singularBeta, ply, cutnode, prevMove, ttMove);
            if (singularScore < singularBeta)
            {
                extension = 1;
            }
        }

        int newDepth = depth - 1 + extension;

        hist.push_back(pos.key);
        int score;
        if (moveCount == 1)
        {
            score = -negamax(child, newDepth, -beta, -alpha, ply + 1, false, move);
        }
        else
        {
            int reduction = 0;
            if (depth >= 3 && moveCount >= 4 && quiet && !inCheck)
            {
                reduction = Reductions[std::min(depth, MAX_PLY - 1)][std::min(moveCount, 63)];
                if (pvNode)
                {
                    reduction--;
                }
                if (cutnode)
                {
                    reduction++;
                }
                reduction = std::clamp(reduction, 0, newDepth - 1);
            }
            score = -negamax(child, newDepth - reduction, -alpha - 1, -alpha, ply + 1, true, move);
            if (score > alpha && reduction > 0)
            {
                score = -negamax(child, newDepth, -alpha - 1, -alpha, ply + 1, !cutnode, move);
            }
            if (score > alpha && score < beta)
            {
                score = -negamax(child, newDepth, -beta, -alpha, ply + 1, false, move);
            }
        }
        hist.pop_back();
        if (g_stop)
        {
            return 0;
        }

        if (quiet && quietCount < 64)
        {
            quiets[quietCount++] = move;
        }

        if (score > bestScore)
        {
            bestScore = score;
            bestMove  = move;
            if (score > alpha)
            {
                alpha = score;
                if (pvNode)
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
                        if (prevPieceTo >= 0)
                        {
                            counterMoves[prevPieceTo] = move;
                        }
                        int  bonus         = std::min(depth * depth, g_params.historyMax);
                        auto apply_gravity = [&](int &entry, int change) {
                            entry += change - entry * std::abs(change) / 16384;
                        };
                        int currentPieceTo = pos.board[move.from()] * 64 + move.to();
                        apply_gravity(history[pos.stm][move.from()][move.to()], bonus);
                        if (prevPieceTo >= 0)
                        {
                            apply_gravity(contHist[prevPieceTo * 768 + currentPieceTo], bonus);
                        }
                        for (int quietIndex = 0; quietIndex < quietCount - 1; quietIndex++)
                        {
                            Move quietMove = quiets[quietIndex];
                            apply_gravity(history[pos.stm][quietMove.from()][quietMove.to()], -bonus);
                            if (prevPieceTo >= 0)
                            {
                                apply_gravity(
                                    contHist[prevPieceTo * 768 + pos.board[quietMove.from()] * 64 + quietMove.to()],
                                    -bonus);
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

    Bound bound = bestScore >= beta ? BOUND_LOWER : (alpha > origAlpha ? BOUND_EXACT : BOUND_UPPER);
    if (excluded.is_none())
    {
        TT.store(pos.key, bestScore, inCheck ? VALUE_NONE : rawEval, depth, bound, bestMove, ply);

        // Update the eval correction: blend in (search score - raw static eval), but only when the score is
        // a trustworthy signal — not in check, not a tactical (capture) best move, not a mate, and the bound
        // does not contradict the direction of the correction.
        if (!inCheck && !is_mate_score(bestScore) && (bestMove.is_none() || !bestMove.is_capture()) &&
            !(bound == BOUND_LOWER && bestScore <= rawEval) && !(bound == BOUND_UPPER && bestScore >= rawEval))
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
        int alpha = -VALUE_INF, beta = VALUE_INF, delta = g_params.aspirationDelta;
        if (depth >= 4)
        {
            alpha = std::max(-VALUE_INF, score - delta);
            beta  = std::min(VALUE_INF, score + delta);
        }
        while (true)
        {
            int windowScore = negamax(root, depth, alpha, beta, 0, false, Move::none());
            if (g_stop)
            {
                break;
            }
            score = windowScore;
            if (windowScore <= alpha)
            {
                beta  = (alpha + beta) / 2;
                alpha = std::max(-VALUE_INF, windowScore - delta);
                delta += delta / 2;
            }
            else if (windowScore >= beta)
            {
                beta = std::min(VALUE_INF, windowScore + delta);
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
            int64_t  elapsedMs      = elapsed();
            uint64_t nodesPerSecond = elapsedMs ? nodes * 1000 / elapsedMs : nodes;
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
                   scoreStr, (unsigned long long)nodes, (unsigned long long)nodesPerSecond, (long long)elapsedMs,
                   TT.hashfull());
            for (int pvIndex = 0; pvIndex < pvLen[0]; pvIndex++)
            {
                printf(" %s", pvTable[0][pvIndex].to_uci().c_str());
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
