#include "datagen.h"
#include "eval.h"
#include "movegen.h"
#include "position.h"
#include "search.h"
#include "tt.h"
#include "types.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace
{

static const char *START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// Datagen adjudication / filtering knobs.
constexpr int MAX_GAME_PLIES   = 400;  // hard cap; unfinished games score as draws
constexpr int WIN_ADJ_SCORE    = 2000; // |cp| threshold to start counting toward a win adjudication
constexpr int WIN_ADJ_PLIES    = 5;    // consecutive plies above threshold -> adjudicate
constexpr int RECORD_SCORE_CAP = 1500; // skip positions already this decided (noise for eval training)

struct Record
{
    std::string fen;
    int         score; // cp, side-to-move POV
    Color       stm;
};

// Local draw test (search's is_draw is private): 50-move, insufficient material, and 2-fold repetition
// over the in-game history (keys of positions *before* `pos`).
bool datagen_is_draw(const Position &pos, const std::vector<uint64_t> &hist)
{
    if (pos.halfmove >= 100)
    {
        return true;
    }
    if (!(pos.byType[PAWN] | pos.byType[ROOK] | pos.byType[QUEEN]))
    {
        int wm = popcount(pos.byColor[WHITE] & (pos.byType[KNIGHT] | pos.byType[BISHOP]));
        int bm = popcount(pos.byColor[BLACK] & (pos.byType[KNIGHT] | pos.byType[BISHOP]));
        if (wm <= 1 && bm <= 1)
        {
            return true;
        }
    }
    int end     = (int)hist.size();
    int stop_at = end - pos.halfmove;
    if (stop_at < 0)
    {
        stop_at = 0;
    }
    for (int i = end - 2; i >= stop_at; i -= 2)
    {
        if (hist[i] == pos.key)
        {
            return true;
        }
    }
    return false;
}

// Play `openingPlies` uniformly-random legal plies from startpos. Returns false if a terminal position is
// hit (caller retries) so every game starts from a legal, non-terminal, varied position.
bool random_opening(Position &pos, std::vector<uint64_t> &hist, std::mt19937_64 &rng, int openingPlies)
{
    pos.set_fen(START_FEN);
    hist.clear();
    for (int i = 0; i < openingPlies; i++)
    {
        MoveList l;
        generate_legal(pos, l);
        if (l.size() == 0)
        {
            return false;
        }
        Move m = l[rng() % l.size()];
        hist.push_back(pos.key);
        pos.ply = 0;
        pos.make_move(m);
    }
    // Reject openings that are already terminal.
    MoveList l;
    generate_legal(pos, l);
    return l.size() != 0;
}

} // namespace

int run_datagen(int argc, char **argv)
{
    // argv: [0]=datagen [1]=games [2]=out [3]=seed [4]=nodes [5]=openingPlies
    if (argc < 3)
    {
        fprintf(stderr, "usage: %s datagen <games> <out.txt> [seed] [nodes] [openingPlies]\n", argv[0]);
        return 1;
    }
    long        games        = atol(argv[1]);
    const char *outPath      = argv[2];
    uint64_t    seed         = argc > 3 ? strtoull(argv[3], nullptr, 10) : 0x9E3779B97F4A7C15ULL;
    int         nodes        = argc > 4 ? atoi(argv[4]) : 5000;
    int         openingPlies = argc > 5 ? atoi(argv[5]) : 8;

    FILE *out = std::fopen(outPath, "w");
    if (!out)
    {
        fprintf(stderr, "datagen: cannot open %s\n", outPath);
        return 1;
    }

    std::mt19937_64 rng(seed);
    Searcher        searcher;
    searcher.silent       = true;
    searcher.moveOverhead = 0;

    SearchLimits lim;
    lim.nodes = nodes;

    uint64_t totalPos = 0;
    long     finished = 0;
    auto     t0       = std::chrono::steady_clock::now();

    std::vector<Record>   pending;
    std::vector<uint64_t> hist;

    for (long g = 0; g < games; g++)
    {
        Position pos;
        while (!random_opening(pos, hist, rng, openingPlies))
        { /* retry until non-terminal */
        }
        TT.clear();

        pending.clear();
        int gameResult  = 0; // +1 white win, -1 black win, 0 draw
        int winAdjCount = 0;
        int adjSide     = 0;

        for (int ply = 0; ply < MAX_GAME_PLIES; ply++)
        {
            MoveList legal;
            generate_legal(pos, legal);
            if (legal.size() == 0)
            {
                gameResult = pos.in_check() ? (pos.stm == WHITE ? -1 : +1) : 0; // mated stm loses
                break;
            }
            if (datagen_is_draw(pos, hist))
            {
                gameResult = 0;
                break;
            }

            pos.ply       = 0;
            searcher.hist = hist;
            Move m        = searcher.go(pos, lim);
            if (m.is_none())
            {
                gameResult = 0;
                break;
            }
            int score = searcher.rootScore; // cp, stm POV

            // Record quiet, not-yet-decided positions (one per ply).
            if (!pos.in_check() && m.is_quiet() && std::abs(score) < RECORD_SCORE_CAP)
            {
                pending.push_back({pos.fen(), score, pos.stm});
            }

            // Win adjudication (white POV).
            int ws   = pos.stm == WHITE ? score : -score;
            int side = ws > WIN_ADJ_SCORE ? +1 : (ws < -WIN_ADJ_SCORE ? -1 : 0);
            if (side != 0 && side == adjSide)
            {
                if (++winAdjCount >= WIN_ADJ_PLIES)
                {
                    gameResult = side;
                    break;
                }
            }
            else
            {
                adjSide     = side;
                winAdjCount = side != 0 ? 1 : 0;
            }

            hist.push_back(pos.key);
            pos.make_move(m);
        }

        // Emit records with the final WDL from each record's side-to-move POV.
        for (const Record &r : pending)
        {
            double wdl = gameResult == 0 ? 0.5 : (((gameResult > 0) == (r.stm == WHITE)) ? 1.0 : 0.0);
            std::fprintf(out, "%s;%d;%.1f\n", r.fen.c_str(), r.score, wdl);
        }
        totalPos += pending.size();
        finished++;

        if (finished % 50 == 0 || g == games - 1)
        {
            std::fflush(out);
            double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            fprintf(stderr, "[seed %llu] games %ld/%ld  positions %llu  %.0f pos/s\n", (unsigned long long)seed,
                    finished, games, (unsigned long long)totalPos, sec > 0 ? totalPos / sec : 0.0);
        }
    }

    std::fclose(out);
    return 0;
}
