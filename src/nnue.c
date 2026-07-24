#include "nnue.h"
#include "movegen.h"
#include "types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

bool nnue_g_loaded = false;

// Architecture / quantisation contract — must match trainer/features.py exactly.
enum
{
    HIDDEN_SIZE       = NNUE_HIDDEN, // 512
    BASE_FEATURES     = 768,         // per-king-bucket block (2 colors x 6 types x 64 squares)
    NUM_KING_BUCKETS  = 8,           // 4 file-pairs x 2 board-halves, keyed on the perspective king
    INPUT_FEATURES    = NUM_KING_BUCKETS * BASE_FEATURES, // 6144 feature-transformer rows
    QUANT_ACCUMULATOR = 255,                              // QA
    QUANT_OUTPUT      = 64,                               // QB
    EVALUATION_SCALE  = 400
};

static const char NNUE_MAGIC[8] = {'Z', 'N', 'N', 'U', 'E', '3', '\0', '\0'};

typedef struct Network
{
    int16_t feature_transformer_weight[INPUT_FEATURES][HIDDEN_SIZE]; // feature-major, king-bucketed
    int16_t feature_transformer_bias[HIDDEN_SIZE];
    int16_t output_weight[2 * HIDDEN_SIZE]; // own half [0,HIDDEN), opponent half [HIDDEN,2*HIDDEN)
    int32_t output_bias;
} Network;

static Network network;

// King-input bucket for a perspective-relative king square: 4 file-pairs x 2 board-halves.
// Must match king_bucket() in trainer/features.py.
static inline int king_bucket(int relative_king_square)
{
    int file_pair = (relative_king_square & 7) / 2;  // 0..3
    int half      = (relative_king_square >> 3) / 4; // 0..1
    return half * 4 + file_pair;                     // 0..7
}

// Perspective-relative king square (Black mirrors vertically), for bucket selection.
static inline int relative_king_square(Color perspective, int king_square)
{
    return (perspective == WHITE) ? king_square : (king_square ^ 56);
}

// Index within a single 768 king-bucket block (no bucket offset). The caller adds bucket * BASE_FEATURES.
static inline int base_feature_index(Color perspective, Color piece_color, PieceType piece_type, int square)
{
    int relative_color  = (piece_color == perspective) ? 0 : 1;
    int relative_square = (perspective == WHITE) ? square : (square ^ 56);
    return relative_color * 384 + piece_type * 64 + relative_square;
}

// Full king-bucketed feature index for `perspective` given its cached king bucket.
static inline int feature_index(int king_bucket_index, Color perspective, Color piece_color, PieceType piece_type,
                                int square)
{
    return king_bucket_index * BASE_FEATURES + base_feature_index(perspective, piece_color, piece_type, square);
}

static inline void add_column(int16_t *accumulator, const int16_t *column)
{
#if defined(__AVX2__)
    for (int i = 0; i < HIDDEN_SIZE; i += 16)
    {
        __m256i acc = _mm256_loadu_si256((const __m256i *)(accumulator + i));
        __m256i col = _mm256_loadu_si256((const __m256i *)(column + i));
        _mm256_storeu_si256((__m256i *)(accumulator + i), _mm256_add_epi16(acc, col));
    }
#else
    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        accumulator[i] += column[i];
    }
#endif
}

static inline void sub_column(int16_t *accumulator, const int16_t *column)
{
#if defined(__AVX2__)
    for (int i = 0; i < HIDDEN_SIZE; i += 16)
    {
        __m256i acc = _mm256_loadu_si256((const __m256i *)(accumulator + i));
        __m256i col = _mm256_loadu_si256((const __m256i *)(column + i));
        _mm256_storeu_si256((__m256i *)(accumulator + i), _mm256_sub_epi16(acc, col));
    }
#else
    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        accumulator[i] -= column[i];
    }
#endif
}

#if !defined(__AVX2__)
static inline int32_t clamp_accumulator(int32_t value)
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
#endif

// Accumulator refresh cache ("finny tables"). Per thread, for each (perspective, king bucket) we keep the
// last accumulator computed for that bucket and the board bitboards it was built from. A king move that
// switches a perspective to bucket B rebuilds by applying only the piece diffs versus that cached board — a
// handful of column ops instead of a full 32-piece rescan. Always correct (the diffs are exact); a net
// generation counter invalidates the whole cache when a new net is loaded. _Thread_local ⇒ Lazy-SMP safe.
static uint32_t g_net_generation = 0;

typedef struct RefreshCacheEntry
{
    _Alignas(32) int16_t values[HIDDEN_SIZE];
    Bitboard by_color[2];
    Bitboard by_type[6];
    uint32_t net_generation; // 0 ⇒ never populated for the current net
} RefreshCacheEntry;

static _Thread_local RefreshCacheEntry g_refresh_cache[2][NUM_KING_BUCKETS];

// ---- incremental accumulator maintenance -------------------------------------------------------------
// Each perspective indexes into its own cached king bucket (accumulator->king_bucket[perspective]); a piece
// add/remove/move within the same king bucket is a pure incremental update. King moves that change a side's
// bucket are handled by make_move via refresh_perspective (the whole perspective shifts blocks).
void nnue_add_feature(NnueAccumulator *accumulator, Color color, PieceType type, int square)
{
    add_column(
        accumulator->values[WHITE],
        network.feature_transformer_weight[feature_index(accumulator->king_bucket[WHITE], WHITE, color, type, square)]);
    add_column(
        accumulator->values[BLACK],
        network.feature_transformer_weight[feature_index(accumulator->king_bucket[BLACK], BLACK, color, type, square)]);
}

void nnue_remove_feature(NnueAccumulator *accumulator, Color color, PieceType type, int square)
{
    sub_column(
        accumulator->values[WHITE],
        network.feature_transformer_weight[feature_index(accumulator->king_bucket[WHITE], WHITE, color, type, square)]);
    sub_column(
        accumulator->values[BLACK],
        network.feature_transformer_weight[feature_index(accumulator->king_bucket[BLACK], BLACK, color, type, square)]);
}

void nnue_move_feature(NnueAccumulator *accumulator, Color color, PieceType type, int from, int to)
{
    for (int perspective = WHITE; perspective <= BLACK; perspective++)
    {
        int bucket = accumulator->king_bucket[perspective];
        sub_column(accumulator->values[perspective],
                   network.feature_transformer_weight[feature_index(bucket, (Color)perspective, color, type, from)]);
        add_column(accumulator->values[perspective],
                   network.feature_transformer_weight[feature_index(bucket, (Color)perspective, color, type, to)]);
    }
}

// Rebuild a single perspective's half (used when that side's king bucket changes, and by refresh()). Uses
// the thread-local refresh cache: start from the cached accumulator for this (perspective, bucket) and apply
// only the piece diffs versus the board it was built from. Cost is proportional to pieces changed, not 32.
void nnue_refresh_perspective(NnueAccumulator *accumulator, const Position *position, Color perspective)
{
    int bucket = king_bucket(relative_king_square(perspective, position_king_sq(position, perspective)));
    accumulator->king_bucket[perspective] = bucket;

    RefreshCacheEntry *cache = &g_refresh_cache[perspective][bucket];
    if (cache->net_generation != g_net_generation)
    {
        memcpy(cache->values, network.feature_transformer_bias, sizeof(cache->values));
        cache->by_color[0] = cache->by_color[1] = 0;
        for (int type = 0; type < 6; type++)
        {
            cache->by_type[type] = 0;
        }
        cache->net_generation = g_net_generation;
    }

    for (int color = WHITE; color <= BLACK; color++)
    {
        for (int type = 0; type < 6; type++)
        {
            Bitboard current = position->by_color[color] & position->by_type[type];
            Bitboard cached  = cache->by_color[color] & cache->by_type[type];
            Bitboard added   = current & ~cached;
            Bitboard removed = cached & ~current;
            while (added)
            {
                int square = pop_lsb(&added);
                add_column(cache->values, network.feature_transformer_weight[feature_index(
                                              bucket, perspective, (Color)color, (PieceType)type, square)]);
            }
            while (removed)
            {
                int square = pop_lsb(&removed);
                sub_column(cache->values, network.feature_transformer_weight[feature_index(
                                              bucket, perspective, (Color)color, (PieceType)type, square)]);
            }
        }
    }
    cache->by_color[WHITE] = position->by_color[WHITE];
    cache->by_color[BLACK] = position->by_color[BLACK];
    for (int type = 0; type < 6; type++)
    {
        cache->by_type[type] = position->by_type[type];
    }
    memcpy(accumulator->values[perspective], cache->values, sizeof(cache->values));
}

void nnue_update_king_bucket(NnueAccumulator *accumulator, const Position *position, Color side)
{
    int new_bucket = king_bucket(relative_king_square(side, position_king_sq(position, side)));
    if (new_bucket != accumulator->king_bucket[side])
    {
        nnue_refresh_perspective(accumulator, position, side);
    }
}

void nnue_refresh(NnueAccumulator *accumulator, const Position *position)
{
    nnue_refresh_perspective(accumulator, position, WHITE);
    nnue_refresh_perspective(accumulator, position, BLACK);
}

// ---- forward pass ------------------------------------------------------------------------------------
#if defined(__AVX2__)
// SCReLU dot for one perspective: sum over i of clamp(acc[i],0,QA)^2 * weight[i], accumulated in int64.
// Bit-identical to the scalar reference: each term is (int64)(clamped*weight)*clamped (the middle product
// is exact in int32 — clamped<=255, |weight|<=32767 => <=8.35M), and integer addition is associative so the
// vector summation order does not matter. QA=255 fits int16, so the clamp is a plain int16 min/max.
static inline int64_t screlu_dot(const int16_t *acc, const int16_t *weight)
{
    const __m256i zero = _mm256_setzero_si256();
    const __m256i qa   = _mm256_set1_epi16((int16_t)QUANT_ACCUMULATOR);
    __m256i       sum  = _mm256_setzero_si256(); // 4 int64 partial sums

    for (int i = 0; i < HIDDEN_SIZE; i += 16)
    {
        __m256i a = _mm256_loadu_si256((const __m256i *)(acc + i));    // 16 int16 accumulator values
        __m256i w = _mm256_loadu_si256((const __m256i *)(weight + i)); // 16 int16 output weights
        __m256i c = _mm256_min_epi16(_mm256_max_epi16(a, zero), qa);   // clamp to [0,255] (16 int16)

        for (int half = 0; half < 2; half++)
        {
            __m128i c128 = half ? _mm256_extracti128_si256(c, 1) : _mm256_castsi256_si128(c);
            __m128i w128 = half ? _mm256_extracti128_si256(w, 1) : _mm256_castsi256_si128(w);
            __m256i c32  = _mm256_cvtepi16_epi32(c128);  // 8 int32 clamped, [0,255]
            __m256i w32  = _mm256_cvtepi16_epi32(w128);  // 8 int32 weights
            __m256i p32  = _mm256_mullo_epi32(c32, w32); // clamped*weight, exact in int32
            // term = clamped * (clamped*weight) as int64. mul_epi32 reads the low 32 bits of each 64-bit lane
            // as a SIGNED int32, so even lanes come from (c32,p32) directly and odd lanes after a 32-bit shift.
            __m256i even = _mm256_mul_epi32(c32, p32);
            __m256i odd  = _mm256_mul_epi32(_mm256_srli_epi64(c32, 32), _mm256_srli_epi64(p32, 32));
            sum          = _mm256_add_epi64(sum, _mm256_add_epi64(even, odd));
        }
    }
    int64_t lanes[4];
    _mm256_storeu_si256((__m256i *)lanes, sum);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3];
}
#endif

int nnue_evaluate(const NnueAccumulator *accumulator, Color stm)
{
    const int16_t *own      = accumulator->values[stm];
    const int16_t *opponent = accumulator->values[color_flip(stm)];
#if defined(__AVX2__)
    int64_t accumulated =
        screlu_dot(own, network.output_weight) + screlu_dot(opponent, network.output_weight + HIDDEN_SIZE);
#else
    int64_t accumulated = 0;
    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        int32_t own_clamped = clamp_accumulator(own[i]);
        accumulated += (int64_t)(own_clamped * network.output_weight[i]) * own_clamped;
        int32_t opponent_clamped = clamp_accumulator(opponent[i]);
        accumulated += (int64_t)(opponent_clamped * network.output_weight[HIDDEN_SIZE + i]) * opponent_clamped;
    }
#endif
    accumulated /= QUANT_ACCUMULATOR;
    accumulated += network.output_bias;
    return (int)(accumulated * EVALUATION_SCALE / (QUANT_ACCUMULATOR * QUANT_OUTPUT));
}

int nnue_evaluate_position(const Position *position)
{
    NnueAccumulator accumulator;
    nnue_refresh(&accumulator, position);
    return nnue_evaluate(&accumulator, position->stm);
}

// ---- loading -----------------------------------------------------------------------------------------
bool nnue_load(const char *path)
{
    FILE *file = fopen(path, "rb");
    if (!file)
    {
        return false;
    }
    char magic[8];
    bool ok = fread(magic, 1, 8, file) == 8;
    if (!ok || memcmp(magic, NNUE_MAGIC, 8) != 0)
    {
        fprintf(stderr, "nnue: bad magic in %s\n", path);
        fclose(file);
        return false;
    }
    // Read into a heap temporary and only commit to the live `network` on a fully-successful read. Reading
    // straight into the global would leave a previously-good net half-overwritten (and still flagged loaded)
    // if the file is truncated — the engine would then evaluate with garbage weights.
    Network *loaded = malloc(sizeof(Network));
    if (loaded == NULL)
    {
        fclose(file);
        fprintf(stderr, "nnue: out of memory loading %s\n", path);
        return false;
    }
    ok = ok && fread(loaded->feature_transformer_weight, 1, sizeof(loaded->feature_transformer_weight), file) ==
                   sizeof(loaded->feature_transformer_weight);
    ok = ok && fread(loaded->feature_transformer_bias, 1, sizeof(loaded->feature_transformer_bias), file) ==
                   sizeof(loaded->feature_transformer_bias);
    ok = ok && fread(loaded->output_weight, 1, sizeof(loaded->output_weight), file) == sizeof(loaded->output_weight);
    ok = ok && fread(&loaded->output_bias, 1, sizeof(loaded->output_bias), file) == sizeof(loaded->output_bias);
    fclose(file);
    if (!ok)
    {
        free(loaded);
        fprintf(stderr, "nnue: truncated file %s\n", path); // previously-loaded net (if any) stays intact
        return false;
    }
    network       = *loaded;
    nnue_g_loaded = true;
    g_net_generation++; // invalidate every thread's refresh cache (weights/bias changed)
    free(loaded);
    return true;
}

// ---- verification helpers ----------------------------------------------------------------------------
int nnue_eval_fens_from_stdin(const char *net_path)
{
    if (!nnue_load(net_path))
    {
        fprintf(stderr, "nnue: failed to load %s\n", net_path);
        return 1;
    }
    char line[512];
    while (fgets(line, sizeof line, stdin) != NULL)
    {
        line[strcspn(line, "\r\n")] = '\0';
        if (line[0] == '\0')
        {
            continue;
        }
        Position position;
        position_init(&position);
        if (!position_set_fen(&position, line))
        {
            printf("0\n"); // malformed FEN: emit a placeholder so output stays line-aligned with the input
            continue;
        }
        printf("%d\n", nnue_evaluate_position(&position));
    }
    return 0;
}

static uint64_t g_check_nodes = 0, g_check_mismatches = 0;
static int      g_check_maxdiff = 0;

static void self_check_walk(Position *position, int depth)
{
    NnueAccumulator fresh;
    nnue_refresh(&fresh, position);
    for (int perspective = 0; perspective < 2; perspective++)
    {
        for (int i = 0; i < NNUE_HIDDEN; i++)
        {
            int diff = abs(position->acc.values[perspective][i] - fresh.values[perspective][i]);
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
    generate_legal(position, &moves, false);
    for (int index = 0; index < moves.count; index++)
    {
        Position child = *position;
        position_make_move(&child, moves.moves[index]);
        self_check_walk(&child, depth - 1);
    }
}

int nnue_run_self_check(const char *net_path)
{
    if (!nnue_load(net_path))
    {
        fprintf(stderr, "nnue: failed to load %s\n", net_path);
        return 1;
    }
    const char *fens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    };
    for (size_t fen_index = 0; fen_index < sizeof(fens) / sizeof(fens[0]); fen_index++)
    {
        Position position;
        position_init(&position);
        position_set_fen(&position, fens[fen_index]);
        self_check_walk(&position, 4);
    }
    printf("nnuecheck: %llu nodes, %llu accumulator mismatches, max|diff|=%d -> %s\n",
           (unsigned long long)g_check_nodes, (unsigned long long)g_check_mismatches, g_check_maxdiff,
           g_check_mismatches ? "FAIL" : "PASS (incremental == refresh)");
    return g_check_mismatches ? 1 : 0;
}
