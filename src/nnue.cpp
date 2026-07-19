#include "nnue.h"
#include "types.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace nnue
{
namespace
{

// Architecture / quantisation contract — must match trainer/features.py exactly.
constexpr int  INPUT_FEATURES    = 768;
constexpr int  HIDDEN_SIZE       = 512;
constexpr int  QUANT_ACCUMULATOR = 255; // QA
constexpr int  QUANT_OUTPUT      = 64;  // QB
constexpr int  EVALUATION_SCALE  = 400;
constexpr char NNUE_MAGIC[8]     = {'Z', 'N', 'N', 'U', 'E', '1', '\0', '\0'};

// Quantised network parameters, laid out exactly as the .nnue file stores them.
struct Network
{
    int16_t feature_transformer_weight[INPUT_FEATURES][HIDDEN_SIZE]; // feature-major
    int16_t feature_transformer_bias[HIDDEN_SIZE];
    int16_t output_weight[2 * HIDDEN_SIZE]; // own half [0,HIDDEN), opponent half [HIDDEN,2*HIDDEN)
    int32_t output_bias;
};

Network network;
bool    network_loaded = false;

// Feature index for one (colour, piece type, square) in `perspective`'s 768-wide input. Side-to-move's
// own pieces occupy [0,384); the opponent's occupy [384,768). Black's perspective mirrors vertically.
inline int feature_index(Color perspective, Color piece_colour, PieceType piece_type, int square)
{
    int relative_colour = (piece_colour == perspective) ? 0 : 1;
    int relative_square = (perspective == WHITE) ? square : (square ^ 56);
    return relative_colour * 384 + piece_type * 64 + relative_square;
}

// Add one feature column into an accumulator (int16, so bit-identical to the scalar/reference path).
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

// Recompute both perspective accumulators from the board (full refresh). int16 accumulators fit these
// nets (verified by the 0 cp gate against the int64 reference).
void refresh_accumulators(const Position &position, int16_t own_accumulator[HIDDEN_SIZE],
                          int16_t opponent_accumulator[HIDDEN_SIZE])
{
    std::memcpy(own_accumulator, network.feature_transformer_bias, sizeof(network.feature_transformer_bias));
    std::memcpy(opponent_accumulator, network.feature_transformer_bias, sizeof(network.feature_transformer_bias));
    Color    side_to_move = position.stm;
    Bitboard occupied     = position.occupied();
    while (occupied)
    {
        int       square       = pop_lsb(occupied);
        Piece     piece        = position.board[square];
        Color     piece_colour = color_of(piece);
        PieceType piece_type   = type_of(piece);
        add_column(own_accumulator,
                   network.feature_transformer_weight[feature_index(side_to_move, piece_colour, piece_type, square)]);
        add_column(opponent_accumulator,
                   network.feature_transformer_weight[feature_index(~side_to_move, piece_colour, piece_type, square)]);
    }
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
    network_loaded = true;
    return true;
}

bool is_loaded()
{
    return network_loaded;
}

int evaluate(const Position &position)
{
    int16_t own_accumulator[HIDDEN_SIZE];
    int16_t opponent_accumulator[HIDDEN_SIZE];
    refresh_accumulators(position, own_accumulator, opponent_accumulator);

    int64_t accumulated = 0;
    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        int32_t own_clamped = clamp_accumulator(own_accumulator[i]);
        accumulated += (int64_t)(own_clamped * network.output_weight[i]) * own_clamped;
        int32_t opponent_clamped = clamp_accumulator(opponent_accumulator[i]);
        accumulated += (int64_t)(opponent_clamped * network.output_weight[HIDDEN_SIZE + i]) * opponent_clamped;
    }
    accumulated /= QUANT_ACCUMULATOR;
    accumulated += network.output_bias;
    return (int)(accumulated * EVALUATION_SCALE / (QUANT_ACCUMULATOR * QUANT_OUTPUT));
}

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

} // namespace nnue
