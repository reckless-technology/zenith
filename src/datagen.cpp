#include "datagen.h"
#include "eval.h"
#include "movegen.h"
#include "nnue.h"
#include "position.h"
#include "search.h"
#include "tt.h"
#include "types.h"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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
bool datagen_is_draw(const Position &pos, const std::vector<uint64_t> &history)
{
    if (pos.halfmove >= 100)
    {
        return true;
    }
    if (!(pos.by_type[PAWN] | pos.by_type[ROOK] | pos.by_type[QUEEN]))
    {
        int white_minors = popcount(pos.by_color[WHITE] & (pos.by_type[KNIGHT] | pos.by_type[BISHOP]));
        int black_minors = popcount(pos.by_color[BLACK] & (pos.by_type[KNIGHT] | pos.by_type[BISHOP]));
        if (white_minors <= 1 && black_minors <= 1)
        {
            return true;
        }
    }
    int history_end = (int)history.size();
    int stop_at     = history_end - pos.halfmove;
    if (stop_at < 0)
    {
        stop_at = 0;
    }
    for (int index = history_end - 2; index >= stop_at; index -= 2)
    {
        if (history[index] == pos.key)
        {
            return true;
        }
    }
    return false;
}

// Load an openings file (EPD/FEN, one position per line) into memory. set_fen ignores trailing EPD
// opcodes, so raw EPD lines work directly. Returns an empty vector if the path is empty/unreadable.
std::vector<std::string> load_opening_book(const std::string &path)
{
    std::vector<std::string> book;
    if (path.empty())
    {
        return book;
    }
    std::ifstream file(path);
    std::string   line;
    while (std::getline(file, line))
    {
        if (!line.empty() && line[0] != '#')
        {
            book.push_back(line);
        }
    }
    return book;
}

// Set up a game start: a book position (if a book is loaded) or the standard start, then `opening_plies`
// uniformly-random legal plies for variety. Returns false if a terminal position is hit (caller retries)
// so every game starts from a legal, non-terminal, varied position.
bool random_opening(Position &pos, std::vector<uint64_t> &history, std::mt19937_64 &rng, int opening_plies,
                    const std::string &start_fen)
{
    pos.set_fen(start_fen);
    history.clear();
    for (int ply_index = 0; ply_index < opening_plies; ply_index++)
    {
        MoveList moves;
        generate_legal(pos, moves);
        if (moves.size() == 0)
        {
            return false;
        }
        Move move = moves[rng() % moves.size()];
        history.push_back(pos.key);
        pos.ply = 0;
        pos.make_move(move);
    }
    // Reject openings that are already terminal.
    MoveList moves;
    generate_legal(pos, moves);
    return moves.size() != 0;
}

} // namespace

int run_datagen(int argc, char **argv)
{
    // argv: [0]=datagen [1]=games [2]=out [3]=seed [4]=nodes [5]=opening_plies
    if (argc < 3)
    {
        fprintf(stderr, "usage: %s datagen <games> <out.txt> [seed] [nodes] [openingPlies] [book.epd] [net.nnue]\n",
                argv[0]);
        return 1;
    }
    long        games         = atol(argv[1]);
    const char *out_path      = argv[2];
    uint64_t    seed          = argc > 3 ? strtoull(argv[3], nullptr, 10) : 0x9E3779B97F4A7C15ULL;
    int         nodes         = argc > 4 ? atoi(argv[4]) : 5000;
    int         opening_plies = argc > 5 ? atoi(argv[5]) : 8;
    std::string book_path     = argc > 6 ? argv[6] : "";
    std::string net_path      = argc > 7 ? argv[7] : "";

    std::vector<std::string> opening_book = load_opening_book(book_path);
    if (!book_path.empty())
    {
        fprintf(stderr, "datagen: loaded %zu opening positions from %s\n", opening_book.size(), book_path.c_str());
    }
    // Optional NNUE net: self-play labels then come from the network (net-in-the-loop), which is how the
    // engine bootstraps above its hand-crafted teacher. Absent => the HCE labels the data.
    if (!net_path.empty())
    {
        fprintf(stderr, "datagen: %s NNUE %s for labels\n", nnue::load(net_path) ? "loaded" : "FAILED to load",
                net_path.c_str());
    }

    FILE *out = std::fopen(out_path, "w");
    if (!out)
    {
        fprintf(stderr, "datagen: cannot open %s\n", out_path);
        return 1;
    }

    std::mt19937_64 rng(seed);
    Searcher        searcher;
    searcher.silent        = true;
    searcher.move_overhead = 0;

    SearchLimits limits;
    limits.nodes = nodes;

    uint64_t total_positions = 0;
    long     finished        = 0;
    auto     start_time      = std::chrono::steady_clock::now();

    std::vector<Record>   pending;
    std::vector<uint64_t> history;

    for (long game_index = 0; game_index < games; game_index++)
    {
        Position           pos;
        const std::string &start_fen =
            opening_book.empty() ? std::string(START_FEN) : opening_book[rng() % opening_book.size()];
        while (!random_opening(pos, history, rng, opening_plies, start_fen))
        { /* retry until non-terminal */
        }
        TT.clear();

        pending.clear();
        int game_result       = 0; // +1 white win, -1 black win, 0 draw
        int win_adj_count     = 0;
        int adjudication_side = 0;

        for (int ply = 0; ply < MAX_GAME_PLIES; ply++)
        {
            MoveList legal;
            generate_legal(pos, legal);
            if (legal.size() == 0)
            {
                game_result = pos.in_check() ? (pos.stm == WHITE ? -1 : +1) : 0; // mated stm loses
                break;
            }
            if (datagen_is_draw(pos, history))
            {
                game_result = 0;
                break;
            }

            pos.ply       = 0;
            searcher.hist = history;
            Move move     = searcher.go(pos, limits);
            if (move.is_none())
            {
                game_result = 0;
                break;
            }
            int score = searcher.root_score; // cp, stm POV

            // Record quiet, not-yet-decided positions (one per ply).
            if (!pos.in_check() && move.is_quiet() && std::abs(score) < RECORD_SCORE_CAP)
            {
                pending.push_back({pos.fen(), score, pos.stm});
            }

            // Win adjudication (white POV).
            int white_score = pos.stm == WHITE ? score : -score;
            int side        = white_score > WIN_ADJ_SCORE ? +1 : (white_score < -WIN_ADJ_SCORE ? -1 : 0);
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

            history.push_back(pos.key);
            pos.make_move(move);
        }

        // Emit records with the final WDL from each record's side-to-move POV.
        for (const Record &record : pending)
        {
            double wdl = game_result == 0 ? 0.5 : (((game_result > 0) == (record.stm == WHITE)) ? 1.0 : 0.0);
            std::fprintf(out, "%s;%d;%.1f\n", record.fen.c_str(), record.score, wdl);
        }
        total_positions += pending.size();
        finished++;

        if (finished % 50 == 0 || game_index == games - 1)
        {
            std::fflush(out);
            double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
            fprintf(stderr, "[seed %llu] games %ld/%ld  positions %llu  %.0f pos/s\n", (unsigned long long)seed,
                    finished, games, (unsigned long long)total_positions,
                    seconds > 0 ? total_positions / seconds : 0.0);
        }
    }

    std::fclose(out);
    return 0;
}

// --- bulletformat -> fen;score;wdl -------------------------------------------------------------------
// bulletformat ChessBoard (32 bytes): occ u64 | pcs[16] (4-bit pieces in occ set-bit order) | score i16
// | result u8 (0/1/2) | ksq u8 | opp_ksq u8 | extra[3]. Board is stored from the side-to-move's POV:
// nibble bit 3 = own(0)/opp(1), bits 0-2 = piece type; square is stm-relative. Emitting "white to move"
// with own=white/opp=black reproduces exactly the stm/ntm feature indices Zenith's engine computes.
int run_bullet2text(int argc, char **argv)
{
    if (argc < 3)
    {
        fprintf(stderr, "usage: %s bullet2text <in.data> <out.txt> [maxRecords] [stride]\n", argv[0]);
        return 1;
    }
    const char *in_path     = argv[1];
    const char *out_path    = argv[2];
    uint64_t    max_records = argc > 3 ? strtoull(argv[3], nullptr, 10) : ~0ULL;
    uint64_t    stride      = argc > 4 ? std::max<uint64_t>(1, strtoull(argv[4], nullptr, 10)) : 1;
    if (max_records == 0)
    {
        max_records = ~0ULL; // 0 means "all records" (with the given stride)
    }

    FILE *in  = std::fopen(in_path, "rb");
    FILE *out = std::fopen(out_path, "w");
    if (!in || !out)
    {
        fprintf(stderr, "bullet2text: cannot open %s / %s\n", in_path, out_path);
        return 1;
    }

    const char   *piece_chars = "PNBRQK";
    unsigned char record[32];
    uint64_t      read_count = 0, written = 0;
    std::string   line;
    while (written < max_records && std::fread(record, 1, 32, in) == 32)
    {
        if ((read_count++ % stride) != 0)
        {
            continue; // subsample the (5.7B-position) dataset for a diverse manageable slice
        }
        uint64_t occupancy;
        int16_t  score;
        std::memcpy(&occupancy, record, 8);
        std::memcpy(&score, record + 24, 2);
        uint8_t result = record[26];

        char board[64];
        std::memset(board, 0, sizeof(board));
        uint64_t occupancy_bits = occupancy;
        int      piece_index    = 0;
        while (occupancy_bits)
        {
            int square = __builtin_ctzll(occupancy_bits);
            occupancy_bits &= occupancy_bits - 1;
            uint8_t nibble = (record[8 + piece_index / 2] >> (4 * (piece_index % 2))) & 0xF;
            piece_index++;
            char piece_char = piece_chars[nibble & 7];
            board[square] = (nibble & 8) ? char(std::tolower(piece_char)) : piece_char; // bit3 set => opponent (black)
        }

        line.clear();
        for (int rank = 7; rank >= 0; rank--)
        {
            int empty_count = 0;
            for (int file = 0; file < 8; file++)
            {
                char square_char = board[rank * 8 + file];
                if (!square_char)
                {
                    empty_count++;
                }
                else
                {
                    if (empty_count)
                    {
                        line += char('0' + empty_count);
                        empty_count = 0;
                    }
                    line += square_char;
                }
            }
            if (empty_count)
            {
                line += char('0' + empty_count);
            }
            if (rank)
            {
                line += '/';
            }
        }
        std::fprintf(out, "%s w - - 0 1;%d;%.1f\n", line.c_str(), (int)score, result / 2.0);
        written++;
    }
    std::fclose(in);
    std::fclose(out);
    fprintf(stderr, "bullet2text: wrote %llu records (stride %llu) to %s\n", (unsigned long long)written,
            (unsigned long long)stride, out_path);
    return 0;
}
