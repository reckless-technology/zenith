#include "nnue.h"
#include "movegen.h"
#include "types.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace nnue
{

bool g_loaded = false;

namespace
{

// Architecture / quantisation contract — must match trainer/features.py exactly.
constexpr int  HIDDEN_SIZE       = NNUE_HIDDEN; // 512
constexpr int  INPUT_FEATURES    = 768;
constexpr int  QUANT_ACCUMULATOR = 255; // QA
constexpr int  QUANT_OUTPUT      = 64;  // QB
constexpr int  EVALUATION_SCALE  = 400;
constexpr char NNUE_MAGIC[8]     = {'Z', 'N', 'N', 'U', 'E', '1', '\0', '\0'};

struct Network
{
    int16_t feature_transformer_weight[INPUT_FEATURES][HIDDEN_SIZE]; // feature-major
    int16_t feature_transformer_bias[HIDDEN_SIZE];
    int16_t output_weight[2 * HIDDEN_SIZE]; // own half [0,HIDDEN), opponent half [HIDDEN,2*HIDDEN)
    int32_t output_bias;
};

Network network;

// Feature index for one (colour, piece type, square) in `perspective`'s 768-wide input.
inline int feature_index(Color perspective, Color piece_colour, PieceType piece_type, int square)
{
    int relative_colour = (piece_colour == perspective) ? 0 : 1;
    int relative_square = (perspective == WHITE) ? square : (square ^ 56);
    return relative_colour * 384 + piece_type * 64 + relative_square;
}

inline void add_column(int16_t *accumulator, const int16_t *column)
{
#if defined(__AVX2__)
    for (int i = 0; i < HIDDEN_SIZE; i += 16)
    {
        __m256i acc = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(accumulator + i));
        __m256i col = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(column + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(accumulator + i), _mm256_add_epi16(acc, col));
    }
#else
    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        accumulator[i] += column[i];
    }
#endif
}

inline void sub_column(int16_t *accumulator, const int16_t *column)
{
#if defined(__AVX2__)
    for (int i = 0; i < HIDDEN_SIZE; i += 16)
    {
        __m256i acc = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(accumulator + i));
        __m256i col = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(column + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(accumulator + i), _mm256_sub_epi16(acc, col));
    }
#else
    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        accumulator[i] -= column[i];
    }
#endif
}

inline int32_t clamp_accumulator(int32_t value)
{
    if (value < 0)
    {
        return 0;
    }
    if (value > QUANT_ACCUMULATOR)
    {
        return QUANT_ACCUMULATOR;
    }
    return value;
}

} // namespace

// ---- incremental accumulator maintenance -------------------------------------------------------------
void add_feature(NnueAccumulator &accumulator, Color colour, PieceType type, int square)
{
    add_column(accumulator.values[WHITE],
               network.feature_transformer_weight[feature_index(WHITE, colour, type, square)]);
    add_column(accumulator.values[BLACK],
               network.feature_transformer_weight[feature_index(BLACK, colour, type, square)]);
}

void remove_feature(NnueAccumulator &accumulator, Color colour, PieceType type, int square)
{
    sub_column(accumulator.values[WHITE],
               network.feature_transformer_weight[feature_index(WHITE, colour, type, square)]);
    sub_column(accumulator.values[BLACK],
               network.feature_transformer_weight[feature_index(BLACK, colour, type, square)]);
}

void move_feature(NnueAccumulator &accumulator, Color colour, PieceType type, int from, int to)
{
    for (int perspective = WHITE; perspective <= BLACK; perspective++)
    {
        sub_column(accumulator.values[perspective],
                   network.feature_transformer_weight[feature_index(Color(perspective), colour, type, from)]);
        add_column(accumulator.values[perspective],
                   network.feature_transformer_weight[feature_index(Color(perspective), colour, type, to)]);
    }
}

void refresh(NnueAccumulator &accumulator, const Position &position)
{
    std::memcpy(accumulator.values[WHITE], network.feature_transformer_bias, sizeof(network.feature_transformer_bias));
    std::memcpy(accumulator.values[BLACK], network.feature_transformer_bias, sizeof(network.feature_transformer_bias));
    Bitboard occupied = position.occupied();
    while (occupied)
    {
        int   square = pop_lsb(occupied);
        Piece piece  = position.board[square];
        add_feature(accumulator, color_of(piece), type_of(piece), square);
    }
}

// ---- forward pass ------------------------------------------------------------------------------------
int evaluate(const NnueAccumulator &accumulator, Color stm)
{
    const int16_t *own         = accumulator.values[stm];
    const int16_t *opponent    = accumulator.values[~stm];
    int64_t        accumulated = 0;
    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        int32_t own_clamped = clamp_accumulator(own[i]);
        accumulated += (int64_t)(own_clamped * network.output_weight[i]) * own_clamped;
        int32_t opponent_clamped = clamp_accumulator(opponent[i]);
        accumulated += (int64_t)(opponent_clamped * network.output_weight[HIDDEN_SIZE + i]) * opponent_clamped;
    }
    accumulated /= QUANT_ACCUMULATOR;
    accumulated += network.output_bias;
    return (int)(accumulated * EVALUATION_SCALE / (QUANT_ACCUMULATOR * QUANT_OUTPUT));
}

int evaluate(const Position &position)
{
    NnueAccumulator accumulator;
    refresh(accumulator, position);
    return evaluate(accumulator, position.stm);
}

// ---- loading -----------------------------------------------------------------------------------------
bool load(const std::string &path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        return false;
    }
    char magic[8];
    file.read(magic, 8);
    if (std::memcmp(magic, NNUE_MAGIC, 8) != 0)
    {
        std::fprintf(stderr, "nnue: bad magic in %s\n", path.c_str());
        return false;
    }
    file.read(reinterpret_cast<char *>(network.feature_transformer_weight), sizeof(network.feature_transformer_weight));
    file.read(reinterpret_cast<char *>(network.feature_transformer_bias), sizeof(network.feature_transformer_bias));
    file.read(reinterpret_cast<char *>(network.output_weight), sizeof(network.output_weight));
    file.read(reinterpret_cast<char *>(&network.output_bias), sizeof(network.output_bias));
    if (!file)
    {
        std::fprintf(stderr, "nnue: truncated file %s\n", path.c_str());
        return false;
    }
    g_loaded = true;
    return true;
}

// ---- verification helpers ----------------------------------------------------------------------------
int eval_fens_from_stdin(const std::string &net_path)
{
    if (!load(net_path))
    {
        std::fprintf(stderr, "nnue: failed to load %s\n", net_path.c_str());
        return 1;
    }
    std::string line;
    while (std::getline(std::cin, line))
    {
        if (line.empty())
        {
            continue;
        }
        Position position;
        position.set_fen(line);
        std::printf("%d\n", evaluate(position));
    }
    return 0;
}

namespace
{
uint64_t g_check_nodes = 0, g_check_mismatches = 0;
int      g_check_maxdiff = 0;

void self_check_walk(Position &position, int depth)
{
    NnueAccumulator fresh;
    refresh(fresh, position);
    for (int perspective = 0; perspective < 2; perspective++)
    {
        for (int i = 0; i < NNUE_HIDDEN; i++)
        {
            int diff = std::abs(position.acc.values[perspective][i] - fresh.values[perspective][i]);
            if (diff)
            {
                g_check_mismatches++;
            }
            if (diff > g_check_maxdiff)
            {
                g_check_maxdiff = diff;
            }
        }
    }
    g_check_nodes++;
    if (depth == 0)
    {
        return;
    }
    MoveList moves;
    generate_legal(position, moves);
    for (Move m : moves)
    {
        Position child = position;
        child.make_move(m);
        self_check_walk(child, depth - 1);
    }
}
} // namespace

int run_self_check(const std::string &net_path)
{
    if (!load(net_path))
    {
        std::fprintf(stderr, "nnue: failed to load %s\n", net_path.c_str());
        return 1;
    }
    const char *fens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    };
    for (const char *fen : fens)
    {
        Position position;
        position.set_fen(fen);
        self_check_walk(position, 4);
    }
    std::printf("nnuecheck: %llu nodes, %llu accumulator mismatches, max|diff|=%d -> %s\n",
                (unsigned long long)g_check_nodes, (unsigned long long)g_check_mismatches, g_check_maxdiff,
                g_check_mismatches ? "FAIL" : "PASS (incremental == refresh)");
    return g_check_mismatches ? 1 : 0;
}

} // namespace nnue
