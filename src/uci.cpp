#include "uci.h"
#include "eval.h"
#include "movegen.h"
#include "position.h"
#include "search.h"
#include "tt.h"
#include <chrono>
#include <functional>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>

static const char* START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

namespace {

Position game;
std::vector<uint64_t> gameHist; // keys of positions before `game`
Searcher searcher;
std::thread searchThread;

void join_search() {
    if (searchThread.joinable()) {
        searcher.stop = true;
        searchThread.join();
    }
}

Move parse_move(const Position& pos, const std::string& s) {
    MoveList l;
    generate_legal(pos, l);
    for (Move m : l)
        if (m.to_uci() == s) return m;
    return Move::none();
}

void set_position(std::istringstream& is) {
    std::string token;
    is >> token;
    Position pos;
    if (token == "startpos") {
        pos.set_fen(START_FEN);
        is >> token; // maybe "moves"
    } else if (token == "fen") {
        std::string fen;
        while (is >> token && token != "moves") fen += token + " ";
        pos.set_fen(fen);
    }
    gameHist.clear();
    if (token == "moves") {
        std::string mv;
        while (is >> mv) {
            Move m = parse_move(pos, mv);
            if (m.is_none()) break;
            gameHist.push_back(pos.key);
            pos.ply = 0;
            pos.make_move(m);
        }
    }
    pos.ply = 0;
    game = pos;
}

void perft_divide(Position& pos, int depth) {
    MoveList l;
    generate_legal(pos, l);
    uint64_t total = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (Move m : l) {
        Position c = pos;
        c.make_move(m);
        // recursive perft
        std::function<uint64_t(Position&, int)> pf = [&](Position& p, int d) -> uint64_t {
            if (d == 0) return 1;
            MoveList ml;
            generate_legal(p, ml);
            if (d == 1) return ml.size();
            uint64_t n = 0;
            for (Move mm : ml) {
                Position cc = p;
                cc.make_move(mm);
                n += pf(cc, d - 1);
            }
            return n;
        };
        uint64_t n = depth == 1 ? 1 : pf(c, depth - 1);
        total += n;
        printf("%s: %llu\n", m.to_uci().c_str(), (unsigned long long)n);
    }
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("\nnodes %llu  time %.2fs  %.1f Mnps\n", (unsigned long long)total, sec, total / sec / 1e6);
}

void go(std::istringstream& is) {
    join_search();
    SearchLimits lim;
    std::string token;
    int perftDepth = 0;
    while (is >> token) {
        if (token == "wtime") is >> lim.time[WHITE];
        else if (token == "btime") is >> lim.time[BLACK];
        else if (token == "winc") is >> lim.inc[WHITE];
        else if (token == "binc") is >> lim.inc[BLACK];
        else if (token == "movestogo") is >> lim.movestogo;
        else if (token == "movetime") is >> lim.movetime;
        else if (token == "depth") is >> lim.depth;
        else if (token == "nodes") is >> lim.nodes;
        else if (token == "infinite") lim.infinite = true;
        else if (token == "perft") is >> perftDepth;
    }
    if (perftDepth > 0) {
        Position p = game;
        perft_divide(p, perftDepth);
        return;
    }
    Position root = game;
    std::vector<uint64_t> hist = gameHist;
    searchThread = std::thread([root, hist, lim]() mutable {
        searcher.hist = hist;
        Move best = searcher.go(root, lim);
        printf("bestmove %s\n", best.is_none() ? "0000" : best.to_uci().c_str());
        fflush(stdout);
    });
}

void set_option(std::istringstream& is) {
    std::string token, name, value;
    is >> token; // "name"
    while (is >> token && token != "value") name += (name.empty() ? "" : " ") + token;
    while (is >> token) value += (value.empty() ? "" : " ") + token;
    auto lower = [](std::string s) {
        for (char& c : s) c = tolower(c);
        return s;
    };
    std::string n = lower(name);
    if (n == "hash") TT.resize(std::stoi(value));
    else if (n == "clear hash") TT.clear();
    else if (n == "move overhead") searcher.moveOverhead = std::stoi(value);
    // Threads / Ponder accepted and ignored in v1.
}

} // namespace

void uci_loop() {
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream is(line);
        std::string token;
        is >> token;
        if (token == "uci") {
            printf("id name Zenith 0.1\n");
            printf("id author Jonny Reckless\n");
            printf("option name Hash type spin default 64 min 1 max 65536\n");
            printf("option name Threads type spin default 1 min 1 max 1\n");
            printf("option name Move Overhead type spin default 20 min 0 max 5000\n");
            printf("option name Clear Hash type button\n");
            printf("uciok\n");
            fflush(stdout);
        } else if (token == "isready") {
            printf("readyok\n");
            fflush(stdout);
        } else if (token == "ucinewgame") {
            join_search();
            TT.clear();
            game.set_fen(START_FEN);
            gameHist.clear();
        } else if (token == "position") {
            set_position(is);
        } else if (token == "go") {
            go(is);
        } else if (token == "stop") {
            searcher.stop = true;
        } else if (token == "setoption") {
            set_option(is);
        } else if (token == "d") {
            printf("%s\n", game.fen().c_str());
            fflush(stdout);
        } else if (token == "quit") {
            join_search();
            break;
        }
    }
    join_search();
}

// --- CLI: bench (fixed-depth node signature) and perft suite ---

static const char* BenchFens[] = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
    "2rq1rk1/pp1bppbp/2np1np1/8/2BNP3/2N1BP2/PPPQ2PP/2KR3R w - - 0 1",
    "8/8/8/8/8/8/6k1/4K2R w K - 0 1",
    "rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP2BPPP/R1BQK2R b KQ - 0 1",
    "r2q1rk1/1p1nbppp/p2pbn2/4p3/4P3/1NN1BP2/PPPQ2PP/R3KB1R w KQ - 0 1",
};

void run_bench(int depth) {
    if (depth <= 0) depth = 13;
    uint64_t total = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (const char* fen : BenchFens) {
        TT.clear();
        Position pos;
        pos.set_fen(fen);
        Searcher s;
        s.moveOverhead = 0;
        SearchLimits lim;
        lim.depth = depth;
        s.hist.clear();
        // silence info by redirecting? keep it; users can ignore. Sum nodes.
        s.go(pos, lim);
        total += s.nodes;
    }
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("%llu nodes %.0f nps\n", (unsigned long long)total, total / (sec > 0 ? sec : 1));
}

struct PerftCase {
    const char* fen;
    int depth;
    uint64_t expect;
};

void run_perft_suite() {
    PerftCase suite[] = {
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 6, 119060324ULL},
        {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 5, 193690690ULL},
        {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 6, 11030083ULL},
        {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 5, 15833292ULL},
        {"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 5, 89941194ULL},
    };
    bool ok = true;
    for (auto& c : suite) {
        Position pos;
        pos.set_fen(c.fen);
        std::function<uint64_t(Position&, int)> pf = [&](Position& p, int d) -> uint64_t {
            MoveList ml;
            generate_legal(p, ml);
            if (d <= 1) return ml.size();
            uint64_t n = 0;
            for (Move m : ml) {
                Position cc = p;
                cc.make_move(m);
                n += pf(cc, d - 1);
            }
            return n;
        };
        uint64_t n = pf(pos, c.depth);
        bool pass = n == c.expect;
        ok &= pass;
        printf("[%s] perft(%d)=%llu want %llu  %s\n", pass ? "PASS" : "FAIL", c.depth,
               (unsigned long long)n, (unsigned long long)c.expect, c.fen);
    }
    printf("%s\n", ok ? "ALL PERFT PASS" : "PERFT FAILURES");
}
