// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief The UCI protocol loop, the Lazy-SMP search coordinator, and the CLI self-test subcommands.
 */
#include "uci.h"
#include "book.h"
#include "eval.h"
#include "movegen.h"
#include "nnue.h"
#include "platform.h"
#include "position.h"
#include "search.h"
#include "testfmt.h"
#include "tt.h"
#include "version.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

static const char *const TOKEN_SEPARATORS = " \t\r\n";

static Position     game;
static uint64_t     game_hist[SEARCH_HIST_CAP]; // keys of positions before `game`
static int          game_hist_count = 0;
static Searcher    *pool            = NULL; // one Searcher per search thread (Lazy SMP), malloc'd
static int          pool_size       = 0;
static zen_thread_t search_thread; // coordinator thread: spawns helpers, runs the main search
static bool         is_search_thread_running = false;
static int          thread_count             = 1;
static int64_t      move_overhead            = 20;
static bool         is_own_book_enabled      = false; // default OFF: testing must stay bookless

/** @brief Stop any in-flight search and join the coordinator thread (safe to call when idle). */
static void join_search(void)
{
    if (is_search_thread_running)
    {
        g_stop = true;
        zen_thread_join(&search_thread);
        is_search_thread_running = false;
    }
}

/** @brief Match UCI move @p text against @p pos's legal moves. @return the move, or MOVE_NONE if none matches. */
static Move parse_move(const Position *pos, const char *text)
{
    Move moves[MAX_MOVES];
    generate_legal(pos, moves, false);
    for (int index = 0; moves[index] != MOVE_NONE; index++)
    {
        char uci_buf[8];
        if (strcmp(move_to_uci(moves[index], uci_buf), text) == 0)
        {
            return moves[index];
        }
    }
    return MOVE_NONE;
}

/** @brief Handle the `position` command: set the board (startpos/fen) and replay any `moves`, rebuilding history. */
static void set_position(char **save_ptr)
{
    const char *token = strtok_r(NULL, TOKEN_SEPARATORS, save_ptr);
    Position    pos;
    position_init(&pos);
    bool is_valid = false;
    if (token != NULL && strcmp(token, "startpos") == 0)
    {
        is_valid = position_set_fen(&pos, START_FEN);
        token    = strtok_r(NULL, TOKEN_SEPARATORS, save_ptr); // maybe "moves"
    }
    else if (token != NULL && strcmp(token, "fen") == 0)
    {
        char   fen[512];
        size_t fen_length = 0;
        fen[0]            = '\0';
        while ((token = strtok_r(NULL, TOKEN_SEPARATORS, save_ptr)) != NULL && strcmp(token, "moves") != 0)
        {
            const size_t token_length = strlen(token);
            if (fen_length + token_length + 2 < sizeof fen)
            {
                memcpy(fen + fen_length, token, token_length);
                fen_length += token_length;
                fen[fen_length++] = ' ';
                fen[fen_length]   = '\0';
            }
        }
        is_valid = position_set_fen(&pos, fen);
    }
    // Ignore a malformed / illegal `position` command rather than searching an inconsistent board — keep the
    // previous game position (standard, forgiving UCI behaviour).
    if (!is_valid)
    {
        return;
    }
    game_hist_count = 0;
    if (token != NULL && strcmp(token, "moves") == 0)
    {
        const char *move_text;
        while ((move_text = strtok_r(NULL, TOKEN_SEPARATORS, save_ptr)) != NULL)
        {
            const Move move = parse_move(&pos, move_text);
            if (move_is_none(move))
            {
                break;
            }
            if (game_hist_count >= SEARCH_HIST_CAP - MAX_PLY)
            {
                break; // leave MAX_PLY headroom so the in-tree hist pushes during search stay in bounds
            }
            game_hist[game_hist_count++] = pos.key;
            position_make_move(&pos, move);
        }
    }
    game = pos;
}

/** @brief Recursive perft node count under @p node to @p remaining_depth (helper for perft_divide). */
static uint64_t perft_recurse(const Position *node, const int remaining_depth)
{
    if (remaining_depth == 0)
    {
        return 1;
    }
    Move      child_moves[MAX_MOVES];
    const int move_count = generate_legal(node, child_moves, false);
    if (remaining_depth == 1)
    {
        return (uint64_t)move_count;
    }
    uint64_t node_count = 0;
    for (int index = 0; child_moves[index] != MOVE_NONE; index++)
    {
        Position grandchild = *node;
        position_make_move(&grandchild, child_moves[index]);
        node_count += perft_recurse(&grandchild, remaining_depth - 1);
    }
    return node_count;
}

/** @brief Perft with a per-root-move breakdown (the UCI `go perft` command); prints each move's node count. */
static void perft_divide(const Position *pos, const int depth)
{
    Move moves[MAX_MOVES];
    generate_legal(pos, moves, false);
    uint64_t      total      = 0;
    const int64_t start_time = platform_now_ms();
    for (int index = 0; moves[index] != MOVE_NONE; index++)
    {
        const Move move  = moves[index];
        Position   child = *pos;
        position_make_move(&child, move);
        const uint64_t node_count = depth == 1 ? 1 : perft_recurse(&child, depth - 1);
        total += node_count;
        char uci_buf[8];
        printf("%s: %llu\n", move_to_uci(move, uci_buf), (unsigned long long)node_count);
    }
    const double seconds = (platform_now_ms() - start_time) / 1000.0;
    printf("\nnodes %llu  time %.2fs  %.1f Mnps\n", (unsigned long long)total, seconds, total / seconds / 1e6);
}

/** @brief The coordinator thread's captured search state (root, limits, pre-root history), passed by pointer. */
typedef struct GoArgs
{
    Position     root;                  ///< root position to search
    SearchLimits limits;                ///< stopping conditions
    uint64_t     hist[SEARCH_HIST_CAP]; ///< pre-root position keys for repetition detection
    int          hist_count;            ///< number of valid entries in @ref hist
} GoArgs;

/** @brief Arguments handed to one Lazy-SMP helper thread. */
typedef struct HelperArgs
{
    Searcher     *searcher; ///< this helper's search state
    const GoArgs *args;     ///< shared root/limits (read-only)
} HelperArgs;

/** @brief Lazy-SMP helper thread entry: search silently, sharing the TT and stopping when the main thread does. */
static int helper_thread_main(void *raw)
{
    const HelperArgs *const helper = raw;
    searcher_go(helper->searcher, helper->args->root, &helper->args->limits, false);
    return 0;
}

/** @brief Search coordinator thread: (re)size the pool, launch helpers, run the main search, print bestmove. */
static int go_thread_main(void *raw)
{
    GoArgs *const args           = raw;
    int           active_threads = thread_count < 1 ? 1 : thread_count;
    if (pool_size != active_threads)
    {
        Searcher *const resized = malloc(active_threads * sizeof(Searcher)); // (re)size the Lazy-SMP thread pool
        if (resized == NULL)
        {
            // Out of memory resizing the pool: keep the existing pool if usable, else give up this search.
            if (pool == NULL || pool_size < 1)
            {
                printf("bestmove 0000\n");
                fflush(stdout);
                free(args);
                return 0;
            }
            active_threads = pool_size; // fall back to the pool we already have
        }
        else
        {
            free(pool);
            pool = resized;
            for (int thread_index = 0; thread_index < active_threads; thread_index++)
            {
                searcher_init(&pool[thread_index]);
            }
            pool_size = active_threads;
        }
    }
    for (int thread_index = 0; thread_index < active_threads; thread_index++)
    {
        // each thread gets its own repetition history + move-overhead
        memcpy(pool[thread_index].hist_keys, args->hist, args->hist_count * sizeof(uint64_t));
        pool[thread_index].hist_count    = args->hist_count;
        pool[thread_index].move_overhead = move_overhead;
    }
    g_stop = false;
    zen_thread_t helpers[256];
    HelperArgs   helper_args[256];
    for (int thread_index = 1; thread_index < active_threads; thread_index++)
    {
        helper_args[thread_index].searcher = &pool[thread_index];
        helper_args[thread_index].args     = args;
        zen_thread_create(&helpers[thread_index], helper_thread_main, &helper_args[thread_index]);
    }
    const Move best = searcher_go(&pool[0], args->root, &args->limits, true); // main thread manages time + prints info
    g_stop          = true; // make sure any still-deepening helper stops
    for (int thread_index = 1; thread_index < active_threads; thread_index++)
    {
        zen_thread_join(&helpers[thread_index]);
    }
    char uci_buf[8];
    printf("bestmove %s\n", move_is_none(best) ? "0000" : move_to_uci(best, uci_buf));
    fflush(stdout);
    free(args);
    return 0;
}

/** @brief Handle the `go` command: parse limits, do perft/book shortcuts, else spawn the search coordinator. */
static void go(char **save_ptr)
{
    join_search();
    SearchLimits limits;
    search_limits_init(&limits);
    const char *token;
    int         perft_depth = 0;
    while ((token = strtok_r(NULL, TOKEN_SEPARATORS, save_ptr)) != NULL)
    {
        if (strcmp(token, "infinite") == 0)
        {
            limits.is_infinite = true;
            continue;
        }
        // Every other keyword takes one numeric argument; unknown tokens are skipped.
        bool does_take_value =
            strcmp(token, "wtime") == 0 || strcmp(token, "btime") == 0 || strcmp(token, "winc") == 0 ||
            strcmp(token, "binc") == 0 || strcmp(token, "movestogo") == 0 || strcmp(token, "movetime") == 0 ||
            strcmp(token, "depth") == 0 || strcmp(token, "nodes") == 0 || strcmp(token, "perft") == 0;
        if (!does_take_value)
        {
            continue;
        }
        const char *const value = strtok_r(NULL, TOKEN_SEPARATORS, save_ptr);
        if (value == NULL)
        {
            break;
        }
        if (strcmp(token, "wtime") == 0)
        {
            limits.time[WHITE]      = strtoll(value, NULL, 10);
            limits.has_time_control = true;
        }
        else if (strcmp(token, "btime") == 0)
        {
            limits.time[BLACK]      = strtoll(value, NULL, 10);
            limits.has_time_control = true;
        }
        else if (strcmp(token, "winc") == 0)
        {
            limits.inc[WHITE]       = strtoll(value, NULL, 10);
            limits.has_time_control = true;
        }
        else if (strcmp(token, "binc") == 0)
        {
            limits.inc[BLACK]       = strtoll(value, NULL, 10);
            limits.has_time_control = true;
        }
        else if (strcmp(token, "movestogo") == 0)
        {
            limits.movestogo        = atoi(value);
            limits.has_time_control = true;
        }
        else if (strcmp(token, "movetime") == 0)
        {
            limits.movetime         = strtoll(value, NULL, 10);
            limits.has_time_control = true;
        }
        else if (strcmp(token, "depth") == 0)
        {
            limits.depth = atoi(value);
        }
        else if (strcmp(token, "nodes") == 0)
        {
            limits.nodes = strtoll(value, NULL, 10);
        }
        else if (strcmp(token, "perft") == 0)
        {
            perft_depth = atoi(value);
        }
    }
    if (perft_depth > 0)
    {
        const Position perft_position = game;
        perft_divide(&perft_position, perft_depth);
        return;
    }
    // Opening book (Polyglot): only for real game searches — never for analysis (infinite) or fixed
    // depth/node test searches, so bench signatures and SPRT harness runs are unaffected even if enabled.
    if (is_own_book_enabled && book_is_loaded() && !limits.is_infinite && limits.depth == 0 && limits.nodes == 0)
    {
        const Move book_move = book_probe(&game);
        if (!move_is_none(book_move))
        {
            char uci_buf[8];
            printf("info string book move\n");
            printf("bestmove %s\n", move_to_uci(book_move, uci_buf));
            fflush(stdout);
            return;
        }
    }

    GoArgs *const args = malloc(sizeof(GoArgs));
    if (args == NULL)
    {
        printf("bestmove 0000\n"); // out of memory: still answer the GUI rather than go silent
        fflush(stdout);
        return;
    }
    args->root   = game;
    args->limits = limits;
    memcpy(args->hist, game_hist, game_hist_count * sizeof(uint64_t));
    args->hist_count = game_hist_count;
    if (zen_thread_create(&search_thread, go_thread_main, args) == 0)
    {
        is_search_thread_running = true;
    }
    else
    {
        free(args);
        printf("bestmove 0000\n"); // thread spawn failed: answer so the GUI isn't left hanging
        fflush(stdout);
    }
}

/** @brief Handle the `setoption` command: parse name/value and apply Hash/Threads/EvalFile/book/SPSA params. */
static void set_option(char **save_ptr)
{
    // setoption may only arrive while the engine is idle (UCI spec). Defensively stop any in-flight search
    // first: Hash/Clear Hash/EvalFile mutate state the Lazy-SMP threads read live (tt_resize frees TT.table;
    // nnue_load overwrites the network), so mutating mid-search would be a use-after-free / data race.
    join_search();
    const char *token = strtok_r(NULL, TOKEN_SEPARATORS, save_ptr); // "name"
    char        name[256], value[256];
    name[0]  = '\0';
    value[0] = '\0';
    while ((token = strtok_r(NULL, TOKEN_SEPARATORS, save_ptr)) != NULL && strcmp(token, "value") != 0)
    {
        if (name[0] != '\0')
        {
            strncat(name, " ", sizeof(name) - strlen(name) - 1);
        }
        strncat(name, token, sizeof(name) - strlen(name) - 1);
    }
    while ((token = strtok_r(NULL, TOKEN_SEPARATORS, save_ptr)) != NULL)
    {
        if (value[0] != '\0')
        {
            strncat(value, " ", sizeof(value) - strlen(value) - 1);
        }
        strncat(value, token, sizeof(value) - strlen(value) - 1);
    }
    char option_name[256];
    for (size_t i = 0;; i++)
    {
        option_name[i] = (char)tolower((unsigned char)name[i]);
        if (name[i] == '\0')
        {
            break;
        }
    }
    if (strcmp(option_name, "hash") == 0)
    {
        tt_resize(atoi(value));
    }
    else if (strcmp(option_name, "clear hash") == 0)
    {
        tt_clear();
        eval_cache_clear();
    }
    else if (strcmp(option_name, "move overhead") == 0)
    {
        move_overhead = atoi(value);
    }
    else if (strcmp(option_name, "threads") == 0)
    {
        const int requested_threads = atoi(value);
        thread_count                = requested_threads < 1 ? 1 : (requested_threads > 256 ? 256 : requested_threads);
    }
    else if (strcmp(option_name, "ownbook") == 0)
    {
        is_own_book_enabled = strcmp(value, "true") == 0 || strcmp(value, "1") == 0;
    }
    else if (strcmp(option_name, "bookfile") == 0)
    {
        if (book_load(value))
        {
            printf("info string loaded book %s\n", value);
        }
        else
        {
            printf("info string failed to load book %s\n", value);
        }
        fflush(stdout);
    }
    else if (strcmp(option_name, "evalfile") == 0)
    {
        if (nnue_load(value))
        {
            eval_cache_clear(); // a new net changes every evaluation
            printf("info string loaded NNUE %s\n", value);
        }
        else
        {
            printf("info string failed to load NNUE %s\n", value);
        }
        fflush(stdout);
    }
    else
    {
        // Tunable search parameters (SPSA): match by original-case name, integer value. A non-numeric value
        // simply skips the call.
        char      *end_ptr = NULL;
        const long parsed  = strtol(value, &end_ptr, 10);
        if (end_ptr != value)
        {
            set_search_param(name, (int)parsed);
        }
    }
    // Ponder accepted and ignored.
}

void uci_loop(void)
{
    // Startup banner (pawnstar-style): version = major.minor.<git commit count>, stamped by the Makefile.
    printf("Zenith %s compiled %s %s\n", ZENITH_VERSION_STRING, __DATE__, __TIME__);
    fflush(stdout);
    position_init(&game);
    position_set_fen(&game, START_FEN); // start from a legal position, so a bare/invalid `go` never searches
                                        // the empty board (king_sq would then do lsb(0))
    static char line[1 << 16];
    while (fgets(line, sizeof line, stdin) != NULL)
    {
        char             *save_ptr = NULL;
        const char *const token    = strtok_r(line, TOKEN_SEPARATORS, &save_ptr);
        if (token == NULL)
        {
            continue;
        }
        if (strcmp(token, "uci") == 0)
        {
            printf("id name Zenith %s\n", ZENITH_VERSION_STRING);
            printf("id author Jonny Reckless\n");
            printf("option name Hash type spin default 64 min 1 max 65536\n");
            printf("option name Threads type spin default 1 min 1 max 256\n");
            printf("option name Move Overhead type spin default 20 min 0 max 5000\n");
            printf("option name Clear Hash type button\n");
            printf("option name EvalFile type string default <none>\n");
            printf("option name OwnBook type check default false\n");
            printf("option name BookFile type string default <none>\n");
            // Tunable search parameters (SPSA); defaults reproduce the shipped engine.
            printf("option name RfpMargin type spin default %d min 20 max 200\n", g_params.rfp_margin);
            printf("option name NmpDivisor type spin default %d min 50 max 600\n", g_params.nmp_divisor);
            printf("option name LmpBase type spin default %d min 1 max 10\n", g_params.lmp_base);
            printf("option name FutilityBase type spin default %d min 0 max 300\n", g_params.futility_base);
            printf("option name FutilityMargin type spin default %d min 30 max 200\n", g_params.futility_margin);
            printf("option name SeeCaptureMargin type spin default %d min 20 max 300\n", g_params.see_capture_margin);
            printf("option name LmrBase type spin default %d min 0 max 200\n", g_params.lmr_base_x100);
            printf("option name LmrDivisor type spin default %d min 100 max 400\n", g_params.lmr_divisor_x100);
            printf("option name SingularMargin type spin default %d min 1 max 8\n", g_params.singular_margin);
            printf("option name AspirationDelta type spin default %d min 5 max 60\n", g_params.aspiration_delta);
            printf("option name HistoryMax type spin default %d min 100 max 1200\n", g_params.history_max);
            printf("uciok\n");
            fflush(stdout);
        }
        else if (strcmp(token, "isready") == 0)
        {
            printf("readyok\n");
            fflush(stdout);
        }
        else if (strcmp(token, "ucinewgame") == 0)
        {
            join_search();
            tt_clear();
            position_set_fen(&game, START_FEN);
            game_hist_count = 0;
        }
        else if (strcmp(token, "position") == 0)
        {
            set_position(&save_ptr);
        }
        else if (strcmp(token, "go") == 0)
        {
            go(&save_ptr);
        }
        else if (strcmp(token, "stop") == 0)
        {
            g_stop = true;
        }
        else if (strcmp(token, "setoption") == 0)
        {
            set_option(&save_ptr);
        }
        else if (strcmp(token, "d") == 0)
        {
            char fen_buf[128];
            printf("%s\n", position_fen(&game, fen_buf));
            fflush(stdout);
        }
        else if (strcmp(token, "quit") == 0)
        {
            join_search();
            break;
        }
    }
    join_search();
}

// --- CLI: bench (fixed-depth node signature) and perft suite ---

#define BENCH_DEPTH 13 ///< the fixed depth at which the per-position node counts below are pinned

/**
 * @brief A bench position and its deterministic search node count at BENCH_DEPTH.
 *
 * The node counts are the signature components: their sum (3,065,743) is the bench node signature that guards
 * search determinism. They are reproducible on every platform — the fixed-depth search is all-integer except
 * the LMR reduction table, which is now seeded by our own portable_log (not libm), so the counts no longer
 * depend on a platform's math library. If a search/eval change intentionally moves the signature, re-run
 * `./build/zenith bench` and update these counts.
 */
typedef struct
{
    const char *fen;
    uint64_t    expected; ///< searcher node count at BENCH_DEPTH
} BenchCase;

static const BenchCase BenchCases[] = {
    {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 265353},
    {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 1033087},
    {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 121528},
    {"r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1", 385875},
    {"2rq1rk1/pp1bppbp/2np1np1/8/2BNP3/2N1BP2/PPPQ2PP/2KR3R w - - 0 1", 360723},
    {"8/8/8/8/8/8/6k1/4K2R w K - 0 1", 60442},
    {"rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP2BPPP/R1BQK2R b KQ - 0 1", 566162},
    {"r2q1rk1/1p1nbppp/p2pbn2/4p3/4P3/1NN1BP2/PPPQ2PP/R3KB1R w KQ - 0 1", 272573},
};

/**
 * @brief Fixed-depth search over the bench positions: one uniform result line per position, then a summary.
 *
 * At BENCH_DEPTH each position's node count is checked against its pinned value — a [PASS]/[FAIL] determinism
 * gate like perft, self-contained (no external signature grep). At any other depth there is no reference, so
 * the lines are informational (node count + nps, no verdict).
 * @return 0 if every checked position matched (or the depth has no reference), 1 on any mismatch.
 */
int run_bench(int depth)
{
    if (depth <= 0)
    {
        depth = BENCH_DEPTH;
    }
    const bool   has_reference = depth == BENCH_DEPTH; // node counts are pinned only at BENCH_DEPTH
    const size_t count         = sizeof BenchCases / sizeof BenchCases[0];

    Searcher *const searcher   = malloc(sizeof(Searcher)); // ~2.4MB — heap, not stack
    uint64_t        total      = 0;
    double          total_secs = 0.0;
    int             passed     = 0;
    for (size_t i = 0; i < count; i++)
    {
        tt_clear();
        Position pos;
        position_init(&pos);
        position_set_fen(&pos, BenchCases[i].fen);
        searcher_init(searcher);        // fresh search state per position
        searcher->is_silent     = true; // suppress per-iteration info; print one clean per-position line below
        searcher->move_overhead = 0;
        SearchLimits limits;
        search_limits_init(&limits);
        limits.depth           = depth;
        const int64_t start_ms = platform_now_ms();
        searcher_go(searcher, pos, &limits, true);
        const double   secs  = (platform_now_ms() - start_ms) / 1000.0;
        const uint64_t nodes = searcher->nodes;
        total += nodes;
        total_secs += secs;
        const double mnps = secs > 0.0 ? nodes / secs / 1e6 : 0.0;
        char         nbuf[27];
        if (has_reference)
        {
            const bool is_pass = nodes == BenchCases[i].expected;
            passed += is_pass;
            test_result(is_pass, "bench  depth %2d  %15s nodes  %6.1f Mnps  %s", depth, u64_commas(nodes, nbuf), mnps,
                        BenchCases[i].fen);
        }
        else
        {
            printf("bench  depth %2d  %15s nodes  %6.1f Mnps  %s\n", depth, u64_commas(nodes, nbuf), mnps,
                   BenchCases[i].fen);
        }
    }
    free(searcher);

    const double mean_mnps = total_secs > 0.0 ? total / total_secs / 1e6 : 0.0;
    char         tbuf[27];
    if (has_reference)
    {
        const bool is_all_pass = passed == (int)count;
        test_result(is_all_pass, "bench  %d/%zu positions pass  signature %s  %8.3fs  mean %.1f Mnps", passed, count,
                    u64_commas(total, tbuf), total_secs, mean_mnps);
        return is_all_pass ? 0 : 1;
    }
    printf("bench  %zu positions  signature %s  %8.3fs  mean %.1f Mnps  (depth %d — no reference)\n", count,
           u64_commas(total, tbuf), total_secs, mean_mnps, depth);
    return 0;
}

/** @brief One perft test case: a position and its known leaf count at a given depth. */
typedef struct PerftCase
{
    const char *fen;      ///< position FEN
    int         depth;    ///< perft depth
    uint64_t    expected; ///< known-correct leaf count at @ref depth
} PerftCase;

/** @brief Plain perft: number of legal-move leaves at depth @p depth — the movegen correctness invariant. */
static uint64_t perft(const Position *pos, const int depth)
{
    Move      moves[MAX_MOVES];
    const int move_count = generate_legal(pos, moves, false);
    if (depth <= 1)
    {
        return (uint64_t)move_count;
    }
    uint64_t node_count = 0;
    for (int index = 0; moves[index] != MOVE_NONE; index++)
    {
        Position child = *pos;
        position_make_move(&child, moves[index]);
        node_count += perft(&child, depth - 1);
    }
    return node_count;
}

// Ethereal "standard.epd" perft suite (github.com/AndyGrant/Ethereal), EPD form "FEN;D1 n;D2 n;...".
// Known-correct reference counts that exhaustively cover castling rights, en passant, promotion, pins and
// discovered checks — the exact edge cases a (pseudo-)legal generator must get right.
static const char *PerftEpd[] = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1;D1 20;D2 400;D3 8902;D4 197281;D5 4865609;D6 119060324",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1;D1 48;D2 2039;D3 97862;D4 4085603;D5 "
    "193690690",
    "4k3/8/8/8/8/8/8/4K2R w K - 0 1;D1 15;D2 66;D3 1197;D4 7059;D5 133987;D6 764643",
    "4k3/8/8/8/8/8/8/R3K3 w Q - 0 1;D1 16;D2 71;D3 1287;D4 7626;D5 145232;D6 846648",
    "4k2r/8/8/8/8/8/8/4K3 w k - 0 1;D1 5;D2 75;D3 459;D4 8290;D5 47635;D6 899442",
    "r3k3/8/8/8/8/8/8/4K3 w q - 0 1;D1 5;D2 80;D3 493;D4 8897;D5 52710;D6 1001523",
    "4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1;D1 26;D2 112;D3 3189;D4 17945;D5 532933;D6 2788982",
    "r3k2r/8/8/8/8/8/8/4K3 w kq - 0 1;D1 5;D2 130;D3 782;D4 22180;D5 118882;D6 3517770",
    "8/8/8/8/8/8/6k1/4K2R w K - 0 1;D1 12;D2 38;D3 564;D4 2219;D5 37735;D6 185867",
    "8/8/8/8/8/8/1k6/R3K3 w Q - 0 1;D1 15;D2 65;D3 1018;D4 4573;D5 80619;D6 413018",
    "4k2r/6K1/8/8/8/8/8/8 w k - 0 1;D1 3;D2 32;D3 134;D4 2073;D5 10485;D6 179869",
    "r3k3/1K6/8/8/8/8/8/8 w q - 0 1;D1 4;D2 49;D3 243;D4 3991;D5 20780;D6 367724",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1;D1 26;D2 568;D3 13744;D4 314346;D5 7594526;D6 179862938",
    "r3k2r/8/8/8/8/8/8/1R2K2R w Kkq - 0 1;D1 25;D2 567;D3 14095;D4 328965;D5 8153719;D6 195629489",
    "r3k2r/8/8/8/8/8/8/2R1K2R w Kkq - 0 1;D1 25;D2 548;D3 13502;D4 312835;D5 7736373;D6 184411439",
    "r3k2r/8/8/8/8/8/8/R3K1R1 w Qkq - 0 1;D1 25;D2 547;D3 13579;D4 316214;D5 7878456;D6 189224276",
    "1r2k2r/8/8/8/8/8/8/R3K2R w KQk - 0 1;D1 26;D2 583;D3 14252;D4 334705;D5 8198901;D6 198328929",
    "2r1k2r/8/8/8/8/8/8/R3K2R w KQk - 0 1;D1 25;D2 560;D3 13592;D4 317324;D5 7710115;D6 185959088",
    "r3k1r1/8/8/8/8/8/8/R3K2R w KQq - 0 1;D1 25;D2 560;D3 13607;D4 320792;D5 7848606;D6 190755813",
    "4k3/8/8/8/8/8/8/4K2R b K - 0 1;D1 5;D2 75;D3 459;D4 8290;D5 47635;D6 899442",
    "4k3/8/8/8/8/8/8/R3K3 b Q - 0 1;D1 5;D2 80;D3 493;D4 8897;D5 52710;D6 1001523",
    "4k2r/8/8/8/8/8/8/4K3 b k - 0 1;D1 15;D2 66;D3 1197;D4 7059;D5 133987;D6 764643",
    "r3k3/8/8/8/8/8/8/4K3 b q - 0 1;D1 16;D2 71;D3 1287;D4 7626;D5 145232;D6 846648",
    "4k3/8/8/8/8/8/8/R3K2R b KQ - 0 1;D1 5;D2 130;D3 782;D4 22180;D5 118882;D6 3517770",
    "r3k2r/8/8/8/8/8/8/4K3 b kq - 0 1;D1 26;D2 112;D3 3189;D4 17945;D5 532933;D6 2788982",
    "8/8/8/8/8/8/6k1/4K2R b K - 0 1;D1 3;D2 32;D3 134;D4 2073;D5 10485;D6 179869",
    "8/8/8/8/8/8/1k6/R3K3 b Q - 0 1;D1 4;D2 49;D3 243;D4 3991;D5 20780;D6 367724",
    "4k2r/6K1/8/8/8/8/8/8 b k - 0 1;D1 12;D2 38;D3 564;D4 2219;D5 37735;D6 185867",
    "r3k3/1K6/8/8/8/8/8/8 b q - 0 1;D1 15;D2 65;D3 1018;D4 4573;D5 80619;D6 413018",
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1;D1 26;D2 568;D3 13744;D4 314346;D5 7594526;D6 179862938",
    "r3k2r/8/8/8/8/8/8/1R2K2R b Kkq - 0 1;D1 26;D2 583;D3 14252;D4 334705;D5 8198901;D6 198328929",
    "r3k2r/8/8/8/8/8/8/2R1K2R b Kkq - 0 1;D1 25;D2 560;D3 13592;D4 317324;D5 7710115;D6 185959088",
    "r3k2r/8/8/8/8/8/8/R3K1R1 b Qkq - 0 1;D1 25;D2 560;D3 13607;D4 320792;D5 7848606;D6 190755813",
    "1r2k2r/8/8/8/8/8/8/R3K2R b KQk - 0 1;D1 25;D2 567;D3 14095;D4 328965;D5 8153719;D6 195629489",
    "2r1k2r/8/8/8/8/8/8/R3K2R b KQk - 0 1;D1 25;D2 548;D3 13502;D4 312835;D5 7736373;D6 184411439",
    "r3k1r1/8/8/8/8/8/8/R3K2R b KQq - 0 1;D1 25;D2 547;D3 13579;D4 316214;D5 7878456;D6 189224276",
    "8/1n4N1/2k5/8/8/5K2/1N4n1/8 w - - 0 1;D1 14;D2 195;D3 2760;D4 38675;D5 570726;D6 8107539",
    "8/1k6/8/5N2/8/4n3/8/2K5 w - - 0 1;D1 11;D2 156;D3 1636;D4 20534;D5 223507;D6 2594412",
    "8/8/4k3/3Nn3/3nN3/4K3/8/8 w - - 0 1;D1 19;D2 289;D3 4442;D4 73584;D5 1198299;D6 19870403",
    "K7/8/2n5/1n6/8/8/8/k6N w - - 0 1;D1 3;D2 51;D3 345;D4 5301;D5 38348;D6 588695",
    "k7/8/2N5/1N6/8/8/8/K6n w - - 0 1;D1 17;D2 54;D3 835;D4 5910;D5 92250;D6 688780",
    "8/1n4N1/2k5/8/8/5K2/1N4n1/8 b - - 0 1;D1 15;D2 193;D3 2816;D4 40039;D5 582642;D6 8503277",
    "8/1k6/8/5N2/8/4n3/8/2K5 b - - 0 1;D1 16;D2 180;D3 2290;D4 24640;D5 288141;D6 3147566",
    "8/8/3K4/3Nn3/3nN3/4k3/8/8 b - - 0 1;D1 4;D2 68;D3 1118;D4 16199;D5 281190;D6 4405103",
    "K7/8/2n5/1n6/8/8/8/k6N b - - 0 1;D1 17;D2 54;D3 835;D4 5910;D5 92250;D6 688780",
    "k7/8/2N5/1N6/8/8/8/K6n b - - 0 1;D1 3;D2 51;D3 345;D4 5301;D5 38348;D6 588695",
    "B6b/8/8/8/2K5/4k3/8/b6B w - - 0 1;D1 17;D2 278;D3 4607;D4 76778;D5 1320507;D6 22823890",
    "8/8/1B6/7b/7k/8/2B1b3/7K w - - 0 1;D1 21;D2 316;D3 5744;D4 93338;D5 1713368;D6 28861171",
    "k7/B7/1B6/1B6/8/8/8/K6b w - - 0 1;D1 21;D2 144;D3 3242;D4 32955;D5 787524;D6 7881673",
    "K7/b7/1b6/1b6/8/8/8/k6B w - - 0 1;D1 7;D2 143;D3 1416;D4 31787;D5 310862;D6 7382896",
    "B6b/8/8/8/2K5/5k2/8/b6B b - - 0 1;D1 6;D2 106;D3 1829;D4 31151;D5 530585;D6 9250746",
    "8/8/1B6/7b/7k/8/2B1b3/7K b - - 0 1;D1 17;D2 309;D3 5133;D4 93603;D5 1591064;D6 29027891",
    "k7/B7/1B6/1B6/8/8/8/K6b b - - 0 1;D1 7;D2 143;D3 1416;D4 31787;D5 310862;D6 7382896",
    "K7/b7/1b6/1b6/8/8/8/k6B b - - 0 1;D1 21;D2 144;D3 3242;D4 32955;D5 787524;D6 7881673",
    "7k/RR6/8/8/8/8/rr6/7K w - - 0 1;D1 19;D2 275;D3 5300;D4 104342;D5 2161211;D6 44956585",
    "R6r/8/8/2K5/5k2/8/8/r6R w - - 0 1;D1 36;D2 1027;D3 29215;D4 771461;D5 20506480;D6 525169084",
    "7k/RR6/8/8/8/8/rr6/7K b - - 0 1;D1 19;D2 275;D3 5300;D4 104342;D5 2161211;D6 44956585",
    "R6r/8/8/2K5/5k2/8/8/r6R b - - 0 1;D1 36;D2 1027;D3 29227;D4 771368;D5 20521342;D6 524966748",
    "6kq/8/8/8/8/8/8/7K w - - 0 1;D1 2;D2 36;D3 143;D4 3637;D5 14893;D6 391507",
    "6KQ/8/8/8/8/8/8/7k b - - 0 1;D1 2;D2 36;D3 143;D4 3637;D5 14893;D6 391507",
    "K7/8/8/3Q4/4q3/8/8/7k w - - 0 1;D1 6;D2 35;D3 495;D4 8349;D5 166741;D6 3370175",
    "6qk/8/8/8/8/8/8/7K b - - 0 1;D1 22;D2 43;D3 1015;D4 4167;D5 105749;D6 419369",
    "K7/8/8/3Q4/4q3/8/8/7k b - - 0 1;D1 6;D2 35;D3 495;D4 8349;D5 166741;D6 3370175",
    "8/8/8/8/8/K7/P7/k7 w - - 0 1;D1 3;D2 7;D3 43;D4 199;D5 1347;D6 6249",
    "8/8/8/8/8/7K/7P/7k w - - 0 1;D1 3;D2 7;D3 43;D4 199;D5 1347;D6 6249",
    "K7/p7/k7/8/8/8/8/8 w - - 0 1;D1 1;D2 3;D3 12;D4 80;D5 342;D6 2343",
    "7K/7p/7k/8/8/8/8/8 w - - 0 1;D1 1;D2 3;D3 12;D4 80;D5 342;D6 2343",
    "8/2k1p3/3pP3/3P2K1/8/8/8/8 w - - 0 1;D1 7;D2 35;D3 210;D4 1091;D5 7028;D6 34834",
    "8/8/8/8/8/K7/P7/k7 b - - 0 1;D1 1;D2 3;D3 12;D4 80;D5 342;D6 2343",
    "8/8/8/8/8/7K/7P/7k b - - 0 1;D1 1;D2 3;D3 12;D4 80;D5 342;D6 2343",
    "K7/p7/k7/8/8/8/8/8 b - - 0 1;D1 3;D2 7;D3 43;D4 199;D5 1347;D6 6249",
    "7K/7p/7k/8/8/8/8/8 b - - 0 1;D1 3;D2 7;D3 43;D4 199;D5 1347;D6 6249",
    "8/2k1p3/3pP3/3P2K1/8/8/8/8 b - - 0 1;D1 5;D2 35;D3 182;D4 1091;D5 5408;D6 34822",
    "8/8/8/8/8/4k3/4P3/4K3 w - - 0 1;D1 2;D2 8;D3 44;D4 282;D5 1814;D6 11848",
    "4k3/4p3/4K3/8/8/8/8/8 b - - 0 1;D1 2;D2 8;D3 44;D4 282;D5 1814;D6 11848",
    "8/8/7k/7p/7P/7K/8/8 w - - 0 1;D1 3;D2 9;D3 57;D4 360;D5 1969;D6 10724",
    "8/8/k7/p7/P7/K7/8/8 w - - 0 1;D1 3;D2 9;D3 57;D4 360;D5 1969;D6 10724",
    "8/8/3k4/3p4/3P4/3K4/8/8 w - - 0 1;D1 5;D2 25;D3 180;D4 1294;D5 8296;D6 53138",
    "8/3k4/3p4/8/3P4/3K4/8/8 w - - 0 1;D1 8;D2 61;D3 483;D4 3213;D5 23599;D6 157093",
    "8/8/3k4/3p4/8/3P4/3K4/8 w - - 0 1;D1 8;D2 61;D3 411;D4 3213;D5 21637;D6 158065",
    "k7/8/3p4/8/3P4/8/8/7K w - - 0 1;D1 4;D2 15;D3 90;D4 534;D5 3450;D6 20960",
    "8/8/7k/7p/7P/7K/8/8 b - - 0 1;D1 3;D2 9;D3 57;D4 360;D5 1969;D6 10724",
    "8/8/k7/p7/P7/K7/8/8 b - - 0 1;D1 3;D2 9;D3 57;D4 360;D5 1969;D6 10724",
    "8/8/3k4/3p4/3P4/3K4/8/8 b - - 0 1;D1 5;D2 25;D3 180;D4 1294;D5 8296;D6 53138",
    "8/3k4/3p4/8/3P4/3K4/8/8 b - - 0 1;D1 8;D2 61;D3 411;D4 3213;D5 21637;D6 158065",
    "8/8/3k4/3p4/8/3P4/3K4/8 b - - 0 1;D1 8;D2 61;D3 483;D4 3213;D5 23599;D6 157093",
    "k7/8/3p4/8/3P4/8/8/7K b - - 0 1;D1 4;D2 15;D3 89;D4 537;D5 3309;D6 21104",
    "7k/3p4/8/8/3P4/8/8/K7 w - - 0 1;D1 4;D2 19;D3 117;D4 720;D5 4661;D6 32191",
    "7k/8/8/3p4/8/8/3P4/K7 w - - 0 1;D1 5;D2 19;D3 116;D4 716;D5 4786;D6 30980",
    "k7/8/8/7p/6P1/8/8/K7 w - - 0 1;D1 5;D2 22;D3 139;D4 877;D5 6112;D6 41874",
    "k7/8/7p/8/8/6P1/8/K7 w - - 0 1;D1 4;D2 16;D3 101;D4 637;D5 4354;D6 29679",
    "k7/8/8/6p1/7P/8/8/K7 w - - 0 1;D1 5;D2 22;D3 139;D4 877;D5 6112;D6 41874",
    "k7/8/6p1/8/8/7P/8/K7 w - - 0 1;D1 4;D2 16;D3 101;D4 637;D5 4354;D6 29679",
    "k7/8/8/3p4/4p3/8/8/7K w - - 0 1;D1 3;D2 15;D3 84;D4 573;D5 3013;D6 22886",
    "k7/8/3p4/8/8/4P3/8/7K w - - 0 1;D1 4;D2 16;D3 101;D4 637;D5 4271;D6 28662",
    "7k/3p4/8/8/3P4/8/8/K7 b - - 0 1;D1 5;D2 19;D3 117;D4 720;D5 5014;D6 32167",
    "7k/8/8/3p4/8/8/3P4/K7 b - - 0 1;D1 4;D2 19;D3 117;D4 712;D5 4658;D6 30749",
    "k7/8/8/7p/6P1/8/8/K7 b - - 0 1;D1 5;D2 22;D3 139;D4 877;D5 6112;D6 41874",
    "k7/8/7p/8/8/6P1/8/K7 b - - 0 1;D1 4;D2 16;D3 101;D4 637;D5 4354;D6 29679",
    "k7/8/8/6p1/7P/8/8/K7 b - - 0 1;D1 5;D2 22;D3 139;D4 877;D5 6112;D6 41874",
    "k7/8/6p1/8/8/7P/8/K7 b - - 0 1;D1 4;D2 16;D3 101;D4 637;D5 4354;D6 29679",
    "k7/8/8/3p4/4p3/8/8/7K b - - 0 1;D1 5;D2 15;D3 102;D4 569;D5 4337;D6 22579",
    "k7/8/3p4/8/8/4P3/8/7K b - - 0 1;D1 4;D2 16;D3 101;D4 637;D5 4271;D6 28662",
    "7k/8/8/p7/1P6/8/8/7K w - - 0 1;D1 5;D2 22;D3 139;D4 877;D5 6112;D6 41874",
    "7k/8/p7/8/8/1P6/8/7K w - - 0 1;D1 4;D2 16;D3 101;D4 637;D5 4354;D6 29679",
    "7k/8/8/1p6/P7/8/8/7K w - - 0 1;D1 5;D2 22;D3 139;D4 877;D5 6112;D6 41874",
    "7k/8/1p6/8/8/P7/8/7K w - - 0 1;D1 4;D2 16;D3 101;D4 637;D5 4354;D6 29679",
    "k7/7p/8/8/8/8/6P1/K7 w - - 0 1;D1 5;D2 25;D3 161;D4 1035;D5 7574;D6 55338",
    "k7/6p1/8/8/8/8/7P/K7 w - - 0 1;D1 5;D2 25;D3 161;D4 1035;D5 7574;D6 55338",
    "3k4/3pp3/8/8/8/8/3PP3/3K4 w - - 0 1;D1 7;D2 49;D3 378;D4 2902;D5 24122;D6 199002",
    "7k/8/8/p7/1P6/8/8/7K b - - 0 1;D1 5;D2 22;D3 139;D4 877;D5 6112;D6 41874",
    "7k/8/p7/8/8/1P6/8/7K b - - 0 1;D1 4;D2 16;D3 101;D4 637;D5 4354;D6 29679",
    "7k/8/8/1p6/P7/8/8/7K b - - 0 1;D1 5;D2 22;D3 139;D4 877;D5 6112;D6 41874",
    "7k/8/1p6/8/8/P7/8/7K b - - 0 1;D1 4;D2 16;D3 101;D4 637;D5 4354;D6 29679",
    "k7/7p/8/8/8/8/6P1/K7 b - - 0 1;D1 5;D2 25;D3 161;D4 1035;D5 7574;D6 55338",
    "k7/6p1/8/8/8/8/7P/K7 b - - 0 1;D1 5;D2 25;D3 161;D4 1035;D5 7574;D6 55338",
    "3k4/3pp3/8/8/8/8/3PP3/3K4 b - - 0 1;D1 7;D2 49;D3 378;D4 2902;D5 24122;D6 199002",
    "8/Pk6/8/8/8/8/6Kp/8 w - - 0 1;D1 11;D2 97;D3 887;D4 8048;D5 90606;D6 1030499",
    "n1n5/1Pk5/8/8/8/8/5Kp1/5N1N w - - 0 1;D1 24;D2 421;D3 7421;D4 124608;D5 2193768;D6 37665329",
    "8/PPPk4/8/8/8/8/4Kppp/8 w - - 0 1;D1 18;D2 270;D3 4699;D4 79355;D5 1533145;D6 28859283",
    "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N w - - 0 1;D1 24;D2 496;D3 9483;D4 182838;D5 3605103;D6 71179139",
    "8/Pk6/8/8/8/8/6Kp/8 b - - 0 1;D1 11;D2 97;D3 887;D4 8048;D5 90606;D6 1030499",
    "n1n5/1Pk5/8/8/8/8/5Kp1/5N1N b - - 0 1;D1 24;D2 421;D3 7421;D4 124608;D5 2193768;D6 37665329",
    "8/PPPk4/8/8/8/8/4Kppp/8 b - - 0 1;D1 18;D2 270;D3 4699;D4 79355;D5 1533145;D6 28859283",
    "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1;D1 24;D2 496;D3 9483;D4 182838;D5 3605103;D6 71179139",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -;D1 48;D2 2039;D3 97862;D4 4085603;D5 193690690",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq -;D1 6;D2 264;D3 9467;D4 422333;D5 15833292;D6 "
    "706045033",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ -;D1 44;D2 1486;D3 62379;D4 2103487;D5 89941194",
};

/** @brief Run + time one perft case, print a uniform result line, and accumulate totals. @return pass. */
static bool perft_report_case(const char *fen, const int depth, const uint64_t expected, uint64_t *total_nodes,
                              double *total_secs)
{
    Position pos;
    position_init(&pos);
    position_set_fen(&pos, fen);
    const int64_t  start_ms = platform_now_ms();
    const uint64_t nodes    = perft(&pos, depth);
    const double   secs     = (platform_now_ms() - start_ms) / 1000.0;
    const bool     is_pass  = nodes == expected;
    *total_nodes += nodes;
    *total_secs += secs;
    char nbuf[27];
    test_result(is_pass, "perft  depth %2d  %15s nodes  %6.1f Mnps  %s", depth, u64_commas(nodes, nbuf),
                secs > 0.0 ? nodes / secs / 1e6 : 0.0, fen);
    return is_pass;
}

int run_perft_suite(void)
{
    const PerftCase suite[] = {
        // The canonical CPW positions 1-5.
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 6, 119060324ULL},
        {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 5, 193690690ULL},
        {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 6, 11030083ULL},
        {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 5, 15833292ULL},
        {"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 5, 89941194ULL},
        {"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 1", 4, 3894594ULL}, // CPW pos 6
        // Sedlak's edge-case "catcher" positions (published counts): en passant that reveals check or is
        // pinned, castling rights, and promotion/underpromotion — the cases naive movegen most often gets wrong.
        {"3k4/3p4/8/K1P4r/8/8/8/8 b - - 0 1", 6, 1134888ULL},         // ep capture reveals a rank check
        {"8/8/4k3/8/2p5/8/B2P2K1/8 w - - 0 1", 6, 1015133ULL},        // ep discovered check
        {"8/8/1k6/2b5/2pP4/8/5K2/8 b - d3 0 1", 6, 1440467ULL},       // ep would expose the king (pinned)
        {"5k2/8/8/8/8/8/8/4K2R w K - 0 1", 6, 661072ULL},             // kingside castling
        {"3k4/8/8/8/8/8/8/R3K3 w Q - 0 1", 6, 803711ULL},             // queenside castling
        {"r3k2r/1b4bq/8/8/8/8/7B/R3K2R w KQkq - 0 1", 4, 1274206ULL}, // all four castles available
        {"2K2r2/4P3/8/8/8/8/8/3k4 w - - 0 1", 6, 3821001ULL},         // promotion under/into check
        {"8/P1k5/K7/8/8/8/8/8 w - - 0 1", 6, 92683ULL},               // promotion near the enemy king
        {"K1k5/8/P7/8/8/8/8/8 w - - 0 1", 6, 2217ULL},                // promotion + stalemate traps
        {"8/k1P5/8/1K6/8/8/8/8 w - - 0 1", 7, 567584ULL},             // deep promotion race
    };
    uint64_t total_nodes = 0;
    double   total_secs  = 0.0;
    int      passed = 0, total = 0;
    bool     is_all_pass = true;
    for (size_t case_index = 0; case_index < sizeof(suite) / sizeof(suite[0]); case_index++)
    {
        const PerftCase *const test_case = &suite[case_index];
        const bool             is_pass =
            perft_report_case(test_case->fen, test_case->depth, test_case->expected, &total_nodes, &total_secs);
        is_all_pass &= is_pass;
        passed += is_pass;
        total++;
    }

    // Ethereal EPD suite: validate each position at the deepest depth whose expected count fits a node
    // budget, so all 128 positions are checked while the whole suite still runs in a few seconds.
    const uint64_t budget = 5000000ULL;
    for (size_t entry_index = 0; entry_index < sizeof(PerftEpd) / sizeof(PerftEpd[0]); entry_index++)
    {
        const char *const line      = PerftEpd[entry_index];
        const char *const semicolon = strchr(line, ';');
        char              fen[128];
        const size_t      fen_length = (size_t)(semicolon - line);
        memcpy(fen, line, fen_length);
        fen[fen_length]        = '\0';
        int      best_depth    = 0;
        uint64_t best_expected = 0;
        for (const char *cursor = semicolon; cursor != NULL;)
        {
            const char *const next = strchr(cursor + 1, ';');
            char              token[64];
            size_t            token_length = next ? (size_t)(next - cursor - 1) : strlen(cursor + 1);
            if (token_length >= sizeof token)
            {
                token_length = sizeof token - 1;
            }
            memcpy(token, cursor + 1, token_length);
            token[token_length] = '\0';
            int                depth;
            unsigned long long count;
            if (sscanf(token, "D%d %llu", &depth, &count) == 2 && count <= budget)
            {
                best_depth    = depth;
                best_expected = count;
            }
            cursor = next;
        }
        if (best_depth == 0)
        {
            continue;
        }
        const bool is_pass = perft_report_case(fen, best_depth, best_expected, &total_nodes, &total_secs);
        is_all_pass &= is_pass;
        passed += is_pass;
        total++;
    }
    char nbuf[27];
    test_result(is_all_pass, "perft  %d/%d cases pass  %s nodes total  %8.3fs  mean %.1f Mnps", passed, total,
                u64_commas(total_nodes, nbuf), total_secs, total_secs > 0.0 ? total_nodes / total_secs / 1e6 : 0.0);
    return is_all_pass ? 0 : 1;
}

// --- CLI: legalcheck — the copy-free position_is_legal must agree with position_is_legal_slow on every pseudo-legal
// move ---
static uint64_t g_legal_nodes = 0, g_legal_mismatches = 0;

/** @brief Recurse to @p depth checking position_is_legal / gives_check_fast against copy-make ground truth. */
static void legal_check_walk(const Position *pos, const int depth)
{
    Move pseudo[MAX_MOVES];
    generate_pseudo(pos, pseudo, false);
    const Bitboard checkers          = pos->checkers; // cached; legalcheck also validates it via position_is_legal
    const Bitboard pinned            = position_pinned_to_king(pos);
    const Bitboard discovered        = position_discovered_check_candidates(pos);
    const int      enemy_king_square = position_king_sq(pos, enemy_of(pos->color_to_move));
    for (int index = 0; pseudo[index] != MOVE_NONE; index++)
    {
        const Move move = pseudo[index];
        if (position_is_legal(pos, move, checkers, pinned) != position_is_legal_slow(pos, move))
        {
            if (g_legal_mismatches < 8)
            {
                char fen_buf[128];
                printf("  MISMATCH fast=%d slow=%d move=%d->%d flag=%d  %s\n",
                       position_is_legal(pos, move, checkers, pinned), position_is_legal_slow(pos, move),
                       move_from(move), move_to(move), move_flag(move), position_fen(pos, fen_buf));
            }
            g_legal_mismatches++;
        }
        // gives_check_fast: for legal QUIET non-castle moves it must equal the copy-make ground truth
        // (castling is allowed to conservatively report true — it is only a pruning guard).
        if (move_is_quiet(move) && !move_is_castle(move) && position_is_legal_slow(pos, move))
        {
            Position child = *pos;
            position_make_move(&child, move);
            const bool is_check_truth = position_is_in_check(&child);
            const bool is_check_fast  = position_gives_check_fast(pos, move, discovered, enemy_king_square);
            if (is_check_fast != is_check_truth)
            {
                if (g_legal_mismatches < 8)
                {
                    char fen_buf[128];
                    printf("  CHECK-MISMATCH fast=%d truth=%d move=%d->%d flag=%d  %s\n", is_check_fast, is_check_truth,
                           move_from(move), move_to(move), move_flag(move), position_fen(pos, fen_buf));
                }
                g_legal_mismatches++;
            }
        }
    }
    g_legal_nodes++;
    if (depth == 0)
    {
        return;
    }
    Move legal[MAX_MOVES];
    generate_legal(pos, legal, false);
    for (int index = 0; legal[index] != MOVE_NONE; index++)
    {
        Position child = *pos;
        position_make_move(&child, legal[index]);
        legal_check_walk(&child, depth - 1);
    }
}

int run_legal_check(void)
{
    // Positions chosen to hammer pins, checks, king moves, castling and en passant (incl. the EP discovered-
    // check case that position_is_legal defers to the slow path).
    const char *const fens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "8/8/3p4/1Pp4r/1K3p1k/8/4P1P1/1R6 w - c6 0 1",
        "B6b/8/8/8/2K5/4k3/8/b6B w - - 0 1",
        "7k/RR6/8/8/8/8/rr6/7K w - - 0 1",
    };
    uint64_t total_nodes = 0, total_mismatches = 0;
    double   total_secs = 0.0;
    for (size_t fen_index = 0; fen_index < sizeof(fens) / sizeof(fens[0]); fen_index++)
    {
        Position pos;
        position_init(&pos);
        position_set_fen(&pos, fens[fen_index]);
        g_legal_nodes = g_legal_mismatches = 0;
        const int64_t start_ms             = platform_now_ms();
        legal_check_walk(&pos, 4);
        const double secs = (platform_now_ms() - start_ms) / 1000.0;
        total_nodes += g_legal_nodes;
        total_mismatches += g_legal_mismatches;
        total_secs += secs;
        char nbuf[27];
        test_result(g_legal_mismatches == 0, "legal  %13s nodes  %3llu mism  %8.3fs  %6.1f Mnps  %s",
                    u64_commas(g_legal_nodes, nbuf), (unsigned long long)g_legal_mismatches, secs,
                    secs > 0.0 ? g_legal_nodes / secs / 1e6 : 0.0, fens[fen_index]);
    }
    char nbuf[27];
    test_result(total_mismatches == 0,
                "legal  %s nodes, %llu mismatches  %8.3fs  %6.1f Mnps  (position_is_legal == position_is_legal_slow)",
                u64_commas(total_nodes, nbuf), (unsigned long long)total_mismatches, total_secs,
                total_secs > 0.0 ? total_nodes / total_secs / 1e6 : 0.0);
    return total_mismatches ? 1 : 0;
}

// Adversarial-input gate: exercises the untrusted-input paths (FEN parsing + movegen/eval on the result)
// with malformed and hostile inputs. Meaningful under an ASan/UBSan build (the CI runs it there) — it must
// reject the malformed FENs, accept the legal ones, and never read/write out of bounds. Guards the memory-
// safety hardening against regressions.
int run_fuzz_check(void)
{
    // Malformed FENs that MUST be rejected (return false) without any out-of-bounds access.
    static const char *const reject_fens[] = {
        "pppppppppppppppppppp/8/8/8/8/8/8/8 w - - 0 1", // over-long rank
        "8/8/8/8/8/8/8/8/8/8/Q7 w - - 0 1",             // too many ranks
        "8/8/8/8/8/8/8/8 w - - 0 1",                    // no kings
        "4k3/8/8/8/8/8/8/8 w - - 0 1",                  // only a black king
        "4K3/8/8/8/8/8/8/8 w - - 0 1",                  // only a white king
        "zzzz w - - 0 1",                               // garbage board
        "",                                             // empty
        "8",                                            // truncated
        "rnbqkbnr/pppppppp w - - 0 1",                  // too few ranks (no kings placed)
    };
    // Legal (or leniently-accepted) positions that MUST be accepted and safely searched. The last two carry a
    // malformed ep field, which the parser safely ignores (position otherwise valid → no-ep). The queen swarm
    // has one king per side (so it is accepted) yet generates > 256 pseudo-legal moves — movegen must cap.
    static const char *const accept_fens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  "8/5pk1/6p1/3K4/8/5PP1/8/8 w - - 0 1",
        "QQQ2QQ1/3Q4/1Q4QQ/Q3Q2Q/Q6Q/Q6Q/Q5Q1/KQQQQQQk w - - 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq z9 0 1", // bad ep, safely ignored
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq !5 0 1", // bad ep char, safely ignored
    };
    int failures = 0, cases = 0;
    for (size_t i = 0; i < sizeof(reject_fens) / sizeof(reject_fens[0]); i++)
    {
        Position pos;
        position_init(&pos);
        const bool is_rejected = !position_set_fen(&pos, reject_fens[i]);
        failures += !is_rejected;
        cases++;
        test_result(is_rejected, "fuzz   reject  %s", reject_fens[i][0] ? reject_fens[i] : "(empty)");
    }
    for (size_t i = 0; i < sizeof(accept_fens) / sizeof(accept_fens[0]); i++)
    {
        Position pos;
        position_init(&pos);
        const bool is_accepted = position_set_fen(&pos, accept_fens[i]);
        failures += !is_accepted;
        cases++;
        if (is_accepted)
        {
            // Exercise the downstream paths that OOB'd before hardening: movegen (buffer cap), the legality
            // oracle, and evaluate() (king_sq / attack tables). Under ASan this catches any residual overrun.
            Move pseudo[MAX_MOVES], legal[MAX_MOVES];
            generate_pseudo(&pos, pseudo, false);
            generate_legal(&pos, legal, false);
            (void)evaluate(&pos);
        }
        test_result(is_accepted, "fuzz   accept  %s", accept_fens[i]);
    }
    test_result(failures == 0, "fuzz   %d/%d inputs handled safely", cases - failures, cases);
    return failures ? 1 : 0;
}

/**
 * @brief SEE unit test: static_exchange_eval on hand-verified capture positions.
 *
 * Each case is a {position, capture move, expected SEE} triple worked out by hand against the engine's
 * piece values (P=100, N=320, B=330, R=500, Q=900). Coverage: free captures, singly-defended captures,
 * equal trades, queen/rook/knight blunders into defended pieces, x-ray batteries on both a file (rear rook)
 * and a diagonal (queen behind bishop), least-valuable-attacker ordering, a high-value defender declining a
 * losing recapture, the king unable to recapture into a guarded square, and en-passant (plain and
 * recaptured). Genuine oracle values (correct chess), not a lock-in of current behaviour.
 * @return 0 if every case matches, 1 otherwise.
 */
int run_see_check(void)
{
    static const struct
    {
        const char *fen;
        const char *move;
        int         expected;
    } cases[] = {
        {"4k3/8/8/4p3/8/8/8/4RK2 w - - 0 1", "e1e5", 100},      // rook takes an undefended pawn
        {"4k3/8/3p4/4p3/8/8/8/4RK2 w - - 0 1", "e1e5", -400},   // rook takes a pawn-defended pawn (100-500)
        {"4k3/8/5p2/4p3/3P4/8/8/4K3 w - - 0 1", "d4e5", 0},     // equal pawn trade (100-100)
        {"4k3/8/8/2p5/3p4/8/8/3QK3 w - - 0 1", "d1d4", -800},   // queen takes a pawn-defended pawn (100-900)
        {"4k3/8/8/4r3/8/8/8/4RK2 w - - 0 1", "e1e5", 500},      // rook takes an undefended rook
        {"4k3/8/8/4q3/8/8/8/4RK2 w - - 0 1", "e1e5", 900},      // rook takes an undefended queen
        {"4k3/8/4p3/3p4/8/8/3R4/3RK3 w - - 0 1", "d2d5", -300}, // x-ray: RxP, pxR, RxP (200-500)
        {"4k3/8/8/3Pp3/8/8/8/4K3 w - e6 0 1", "d5e6", 100},     // en-passant capture, undefended
        {"4k3/8/8/3n4/4P3/8/8/4K3 w - - 0 1", "e4d5", 320},     // pawn takes an undefended knight
        {"4k3/8/2p5/3b4/4P3/8/8/4K3 w - - 0 1", "e4d5", 230},   // pawn x bishop, pawn recaptures -> win B for P
        {"4k3/2p5/3p4/8/4N3/8/8/4K3 w - - 0 1", "e4d6", -220},  // knight x pawn, pawn recaptures -> lose N for P
        {"4k3/8/3p4/4n4/8/8/8/4R1K1 w - - 0 1", "e1e5", -180},  // rook x pawn-defended knight -> lose the exchange
        {"4k3/8/3p4/4n4/8/8/4R3/4R1K1 w - - 0 1", "e2e5", -80}, // x-ray file: rook-behind-rook recaptures the pawn
        {"4k3/8/5p2/4p3/8/8/1B6/Q5K1 w - - 0 1", "b2e5", -130}, // x-ray diagonal: queen behind bishop recaptures
        {"4k3/2p5/3b4/8/4N3/8/8/6K1 w - - 0 1", "e4d6", 10},    // knight x bishop, pawn recaptures -> win B for N (+10)
        {"4k3/5p2/8/3Pp3/8/8/8/4K3 w - e6 0 1", "d5e6", 0},     // en passant, then recaptured by a pawn -> even
        {"3qk3/8/8/3b4/2P1P3/8/8/4K3 w - - 0 1", "e4d5", 330},  // pawn x bishop; black queen declines (c4 pawn deters)
        {"8/8/2k5/3p4/4P3/8/8/3R2K1 w - - 0 1", "e4d5", 100},   // king cannot recapture into the rook's guard
        {"4k3/8/4p3/3r4/8/8/6B1/6K1 w - - 0 1", "g2d5", 170},   // bishop x rook, pawn recaptures -> win the exchange
        {"4k3/8/4p3/3n4/8/8/8/3QK3 w - - 0 1", "d1d5", -580},   // queen x pawn-defended knight -> blunder (320-900)
        {"3rk3/8/2p5/3n4/5N2/8/8/3R2K1 w - - 0 1", "f4d5",
         0}, // LVA: pawn (not rook) recaptures first, then RxP=RxR -> 0
    };

    const int case_count = (int)(sizeof cases / sizeof cases[0]);
    int       failures   = 0;
    for (int i = 0; i < case_count; i++)
    {
        Position pos;
        position_init(&pos);
        if (!position_set_fen(&pos, cases[i].fen))
        {
            test_result(false, "see    bad FEN  %s", cases[i].fen);
            failures++;
            continue;
        }
        const Move move = parse_move(&pos, cases[i].move);
        if (move_is_none(move))
        {
            test_result(false, "see    illegal move %s  %s", cases[i].move, cases[i].fen);
            failures++;
            continue;
        }
        const int  see     = static_exchange_eval(&pos, move);
        const bool is_pass = see == cases[i].expected;
        failures += !is_pass;
        test_result(is_pass, "see    %-5s  see %5d  want %5d  %s", cases[i].move, see, cases[i].expected, cases[i].fen);
    }
    test_result(failures == 0, "see    %d/%d cases pass", case_count - failures, case_count);
    return failures ? 1 : 0;
}
