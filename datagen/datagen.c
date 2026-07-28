// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief Self-play datagen and bulletformat conversion: game loop, adjudication, and record emission.
 */
#include "datagen.h"
#include "eval.h"
#include "movegen.h"
#include "nnue.h"
#include "platform.h"
#include "position.h"
#include "search.h"
#include "tt.h"
#include "types.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/** @brief Datagen adjudication / filtering knobs. */
enum
{
    MAX_GAME_PLIES   = 400,  ///< hard cap; unfinished games score as draws
    WIN_ADJ_SCORE    = 2000, ///< |cp| threshold to start counting toward a win adjudication
    WIN_ADJ_PLIES    = 5,    ///< consecutive plies above threshold -> adjudicate
    RECORD_SCORE_CAP = 1500  ///< skip positions already this decided (noise for eval training)
};

/** @brief One pending training record, held until the game's WDL result is known. */
typedef struct Record
{
    char  fen[128]; ///< position FEN
    int   score;    ///< cp, side-to-move POV
    Color stm;      ///< side to move (fixes the WDL POV)
} Record;

// The same xorshift64* PRNG the magic-bitboard generator uses, for opening/move randomisation. Datagen's
// move choices are not covered by any gate (only the data format and game logic matter), so a fast,
// self-contained generator is all that is needed here.
typedef struct DatagenRng
{
    uint64_t state;
} DatagenRng;

static uint64_t rng_next(DatagenRng *rng)
{
    rng->state ^= rng->state >> 12;
    rng->state ^= rng->state << 25;
    rng->state ^= rng->state >> 27;
    return rng->state * 0x2545F4914F6CDD1DULL;
}

/**
 * @brief Load an openings file (EPD/FEN, one position per line) into memory.
 *
 * set_fen ignores trailing EPD opcodes, so raw EPD lines work directly.
 * @return an array of @p count strings, or NULL (count 0) if the path is empty/unreadable.
 */
static char **load_opening_book(const char *path, size_t *count)
{
    *count = 0;
    if (path == NULL || path[0] == '\0')
    {
        return NULL;
    }
    FILE *const file = fopen(path, "r");
    if (!file)
    {
        return NULL;
    }
    char **book     = NULL;
    size_t capacity = 0;
    char   line[512];
    while (fgets(line, sizeof line, file) != NULL)
    {
        line[strcspn(line, "\r\n")] = '\0';
        if (line[0] != '\0' && line[0] != '#')
        {
            if (*count == capacity)
            {
                capacity             = capacity ? capacity * 2 : 1024;
                char **const resized = realloc(book, capacity * sizeof(char *));
                if (resized == NULL)
                {
                    break; // out of memory: keep what we have (realloc left `book` valid)
                }
                book = resized;
            }
            char *const copy = strdup(line);
            if (copy == NULL)
            {
                break;
            }
            book[(*count)++] = copy;
        }
    }
    fclose(file);
    return book;
}

/**
 * @brief Set up a game start: @p start_fen, then @p opening_plies uniformly-random legal plies for variety.
 *
 * @return false if a terminal position is hit (the caller retries), so every game starts from a legal,
 *         non-terminal, varied position.
 */
static bool random_opening(Position *pos, uint64_t *history, int *history_count, DatagenRng *rng,
                           const int opening_plies, const char *start_fen)
{
    if (!position_set_fen(pos, start_fen))
    {
        return false; // malformed opening (bad book/EPD line) — caller retries with the next opening
    }
    *history_count = 0;
    for (int ply_index = 0; ply_index < opening_plies; ply_index++)
    {
        Move      moves[MAX_MOVES];
        const int count = generate_legal(pos, moves, false);
        if (count == 0)
        {
            return false;
        }
        const Move move             = moves[rng_next(rng) % count];
        history[(*history_count)++] = pos->key;
        position_make_move(pos, move);
    }
    // Reject openings that are already terminal.
    Move moves[MAX_MOVES];
    return generate_legal(pos, moves, false) != 0;
}

int run_datagen(Engine *engine, const int argc, char **argv)
{
    // argv: [0]=program [1]=games [2]=out [3]=seed [4]=nodes [5]=opening_plies
    if (argc < 3)
    {
        fprintf(stderr, "usage: %s <games> <out.txt> [seed] [nodes] [openingPlies] [book.epd] [net.nnue]\n", argv[0]);
        return 1;
    }
    const long        games    = atol(argv[1]);
    const char *const out_path = argv[2];
    const uint64_t    seed     = argc > 3 ? strtoull(argv[3], NULL, 10) : 0x9E3779B97F4A7C15ULL;
    const int         nodes    = argc > 4 ? atoi(argv[4]) : 5000;
    // Clamp opening_plies to the history buffer: random_opening writes one history[] entry per ply, and the
    // buffer is history[SEARCH_HIST_CAP]. Leave headroom so the game itself can still record moves afterwards.
    int opening_plies = argc > 5 ? atoi(argv[5]) : 8;
    if (opening_plies < 0)
    {
        opening_plies = 0;
    }
    if (opening_plies > SEARCH_HIST_CAP / 2)
    {
        opening_plies = SEARCH_HIST_CAP / 2;
    }
    const char *const book_path = argc > 6 ? argv[6] : "";
    const char *const net_path  = argc > 7 ? argv[7] : "";

    size_t       opening_book_count = 0;
    char **const opening_book       = load_opening_book(book_path, &opening_book_count);
    if (book_path[0] != '\0')
    {
        fprintf(stderr, "datagen: loaded %zu opening positions from %s\n", opening_book_count, book_path);
    }
    // Optional NNUE net: self-play labels then come from the network (net-in-the-loop), which is how the
    // engine bootstraps above its hand-crafted teacher. Absent => the HCE labels the data.
    if (net_path[0] != '\0')
    {
        engine->net = nnue_load(net_path);
        fprintf(stderr, "datagen: %s NNUE %s for labels\n", engine->net ? "loaded" : "FAILED to load", net_path);
    }

    FILE *const out = fopen(out_path, "w");
    if (!out)
    {
        fprintf(stderr, "datagen: cannot open %s\n", out_path);
        return 1;
    }

    DatagenRng      rng      = {seed ? seed : 0x9E3779B97F4A7C15ULL}; // xorshift must not start at 0
    Searcher *const searcher = malloc(sizeof(Searcher));
    searcher_init(searcher, engine);
    searcher->is_silent     = true;
    searcher->move_overhead = 0;

    SearchLimits limits;
    search_limits_init(&limits);
    limits.nodes = nodes;

    uint64_t      total_positions = 0;
    long          finished        = 0;
    const int64_t start_time      = platform_now_ms();

    static Record   pending[MAX_GAME_PLIES];
    int             pending_count = 0;
    static uint64_t history[SEARCH_HIST_CAP];
    int             history_count = 0;

    for (long game_index = 0; game_index < games; game_index++)
    {
        Position pos;
        position_init(&pos, engine->net);
        const char *const start_fen =
            opening_book_count == 0 ? START_FEN : opening_book[rng_next(&rng) % opening_book_count];
        while (!random_opening(&pos, history, &history_count, &rng, opening_plies, start_fen))
        { /* retry until non-terminal */
        }
        tt_clear(&engine->tt);

        pending_count         = 0;
        int game_result       = 0; // +1 white win, -1 black win, 0 draw
        int win_adj_count     = 0;
        int adjudication_side = 0;

        for (int ply = 0; ply < MAX_GAME_PLIES; ply++)
        {
            Move legal[MAX_MOVES];
            if (generate_legal(&pos, legal, false) == 0)
            {
                game_result = (pos.checkers != 0) ? (pos.color_to_move == WHITE ? -1 : +1) : 0; // mated stm loses
                break;
            }
            if (position_is_draw(&pos, history, history_count))
            {
                game_result = 0;
                break;
            }

            memcpy(searcher->hist_keys, history, history_count * sizeof(uint64_t));
            searcher->hist_count = history_count;
            const Move move      = searcher_go(searcher, pos, &limits, true);
            if (move_is_none(move))
            {
                game_result = 0;
                break;
            }
            const int score = searcher->root_score; // cp, stm POV

            // Record quiet, not-yet-decided positions (one per ply).
            if (!(pos.checkers != 0) && move_is_quiet(move) && abs(score) < RECORD_SCORE_CAP)
            {
                Record *const record = &pending[pending_count++];
                position_fen(&pos, record->fen);
                record->score = score;
                record->stm   = pos.color_to_move;
            }

            // Win adjudication (white POV).
            const int white_score = pos.color_to_move == WHITE ? score : -score;
            const int side        = white_score > WIN_ADJ_SCORE ? +1 : (white_score < -WIN_ADJ_SCORE ? -1 : 0);
            if (side != 0 && side == adjudication_side)
            {
                if (++win_adj_count >= WIN_ADJ_PLIES)
                {
                    game_result = side;
                    break;
                }
            }
            else
            {
                adjudication_side = side;
                win_adj_count     = side != 0 ? 1 : 0;
            }

            history[history_count++] = pos.key;
            position_make_move(&pos, move);
        }

        // Emit records with the final WDL from each record's side-to-move POV.
        for (int record_index = 0; record_index < pending_count; record_index++)
        {
            const Record *const record = &pending[record_index];
            const double wdl = game_result == 0 ? 0.5 : (((game_result > 0) == (record->stm == WHITE)) ? 1.0 : 0.0);
            fprintf(out, "%s;%d;%.1f\n", record->fen, record->score, wdl);
        }
        total_positions += pending_count;
        finished++;

        if (finished % 50 == 0 || game_index == games - 1)
        {
            fflush(out);
            const double seconds = (platform_now_ms() - start_time) / 1000.0;
            fprintf(stderr, "[seed %llu] games %ld/%ld  positions %llu  %.0f pos/s\n", (unsigned long long)seed,
                    finished, games, (unsigned long long)total_positions,
                    seconds > 0 ? total_positions / seconds : 0.0);
        }
    }

    fclose(out);
    free(searcher);
    for (size_t book_index = 0; book_index < opening_book_count; book_index++)
    {
        free(opening_book[book_index]);
    }
    free(opening_book);
    return 0;
}

// --- bulletformat -> fen;score;wdl -------------------------------------------------------------------
// bulletformat ChessBoard (32 bytes): occ u64 | pcs[16] (4-bit pieces in occ set-bit order) | score i16
// | result u8 (0/1/2) | ksq u8 | opp_ksq u8 | extra[3]. Board is stored from the side-to-move's POV:
// nibble bit 3 = own(0)/opp(1), bits 0-2 = piece type; square is stm-relative. Emitting "white to move"
// with own=white/opp=black reproduces exactly the stm/ntm feature indices Zenith's engine computes.
int run_bullet2text(const int argc, char **argv)
{
    if (argc < 3)
    {
        fprintf(stderr, "usage: %s <in.data> <out.txt> [maxRecords] [stride]\n", argv[0]);
        return 1;
    }
    const char *const in_path     = argv[1];
    const char *const out_path    = argv[2];
    uint64_t          max_records = argc > 3 ? strtoull(argv[3], NULL, 10) : ~0ULL;
    uint64_t          stride      = 1;
    if (argc > 4)
    {
        stride = strtoull(argv[4], NULL, 10);
        if (stride < 1)
        {
            stride = 1;
        }
    }
    if (max_records == 0)
    {
        max_records = ~0ULL; // 0 means "all records" (with the given stride)
    }

    FILE *const in  = fopen(in_path, "rb");
    FILE *const out = fopen(out_path, "w");
    if (!in || !out)
    {
        fprintf(stderr, "bullet2text: cannot open %s / %s\n", in_path, out_path);
        return 1;
    }

    const char *const piece_chars = "PNBRQK";
    unsigned char     record[32];
    uint64_t          read_count = 0, written = 0;
    char              line[128];
    while (written < max_records && fread(record, 1, 32, in) == 32)
    {
        if ((read_count++ % stride) != 0)
        {
            continue; // subsample the (5.7B-position) dataset for a diverse manageable slice
        }
        uint64_t occupancy;
        int16_t  score;
        memcpy(&occupancy, record, 8);
        memcpy(&score, record + 24, 2);
        const uint8_t result = record[26];

        char board[64];
        memset(board, 0, sizeof(board));
        uint64_t occupancy_bits = occupancy;
        int      piece_index    = 0;
        while (occupancy_bits)
        {
            const int square = __builtin_ctzll(occupancy_bits);
            occupancy_bits &= occupancy_bits - 1;
            const uint8_t nibble = (record[8 + piece_index / 2] >> (4 * (piece_index % 2))) & 0xF;
            piece_index++;
            const char piece_char = piece_chars[nibble & 7];
            board[square] = (nibble & 8) ? (char)tolower(piece_char) : piece_char; // bit3 set => opponent (black)
        }

        int length = 0;
        for (int rank = 7; rank >= 0; rank--)
        {
            int empty_count = 0;
            for (int file = 0; file < 8; file++)
            {
                const char square_char = board[rank * 8 + file];
                if (!square_char)
                {
                    empty_count++;
                }
                else
                {
                    if (empty_count)
                    {
                        line[length++] = (char)('0' + empty_count);
                        empty_count    = 0;
                    }
                    line[length++] = square_char;
                }
            }
            if (empty_count)
            {
                line[length++] = (char)('0' + empty_count);
            }
            if (rank)
            {
                line[length++] = '/';
            }
        }
        line[length] = '\0';
        fprintf(out, "%s w - - 0 1;%d;%.1f\n", line, (int)score, result / 2.0);
        written++;
    }
    fclose(in);
    fclose(out);
    fprintf(stderr, "bullet2text: wrote %llu records (stride %llu) to %s\n", (unsigned long long)written,
            (unsigned long long)stride, out_path);
    return 0;
}
