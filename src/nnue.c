// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief NNUE internals: king-bucketed feature indexing, incremental accumulator maintenance, and the integer forward.
 */
#include "nnue.h"
#include "movegen.h"
#include "platform.h"
#include "testfmt.h"
#include "types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/** @brief Architecture / quantisation contract — must match trainer/features.py exactly. */
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

/** @brief The quantised network weights, loaded from a `.nnue` file (completes accumulator.h's forward decl). */
struct NnueNetwork
{
    int16_t feature_transformer_weight[INPUT_FEATURES][HIDDEN_SIZE]; ///< feature-major, king-bucketed
    int16_t feature_transformer_bias[HIDDEN_SIZE];                   ///< feature-transformer bias
    int16_t output_weight[2 * HIDDEN_SIZE]; ///< own half [0,HIDDEN), opponent half [HIDDEN,2*HIDDEN)
    int32_t output_bias;                    ///< output bias
};

_Static_assert(NNUE_KING_BUCKETS == NUM_KING_BUCKETS, "header/implementation bucket counts must agree");

/**
 * @brief King-input bucket for a perspective-relative king square: 4 file-pairs x 2 board-halves.
 *
 * Must match king_bucket() in trainer/features.py.
 */
static inline int king_bucket(const int relative_king_square)
{
    const int file_pair = (relative_king_square & 7) / 2;  // 0..3
    const int half      = (relative_king_square >> 3) / 4; // 0..1
    return half * 4 + file_pair;                           // 0..7
}

/** @brief Perspective-relative king square (Black mirrors vertically), for bucket selection. */
static inline int relative_king_square(const Color perspective, const int king_square)
{
    return (perspective == WHITE) ? king_square : (king_square ^ 56);
}

/** @brief Feature index within one 768 king-bucket block; the caller adds bucket * BASE_FEATURES. */
static inline int base_feature_index(const Color perspective, const Color piece_color, const Piece piece,
                                     const int square)
{
    const int relative_color  = (piece_color == perspective) ? 0 : 1;
    const int relative_square = (perspective == WHITE) ? square : (square ^ 56);
    return relative_color * 384 + ((int)piece - 1) * 64 + relative_square; // net features are 0-based types
}

/** @brief Full king-bucketed feature index for @p perspective given its cached king bucket. */
static inline int feature_index(const int king_bucket_index, const Color perspective, const Color piece_color,
                                const Piece piece, const int square)
{
    return king_bucket_index * BASE_FEATURES + base_feature_index(perspective, piece_color, piece, square);
}

/** @brief Add a feature-transformer weight column into an accumulator half (AVX2 or scalar). */
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

/** @brief Subtract a feature-transformer weight column from an accumulator half (AVX2 or scalar). */
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
/** @brief Clamp an accumulator value to [0, QA] for the scalar SCReLU path. */
static inline int32_t clamp_accumulator(const int32_t value)
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

// ---- incremental accumulator maintenance -------------------------------------------------------------
// Each perspective indexes into its own cached king bucket (accumulator->king_bucket[perspective]); a piece
// add/remove/move within the same king bucket is a pure incremental update. King moves that change a side's
// bucket are handled by make_move via refresh_perspective (the whole perspective shifts blocks).
void nnue_add_feature(NnueAccumulator *accumulator, const Color color, const Piece type, const int square)
{
    const NnueNetwork *const net = accumulator->net;
    add_column(
        accumulator->values[WHITE],
        net->feature_transformer_weight[feature_index(accumulator->king_bucket[WHITE], WHITE, color, type, square)]);
    add_column(
        accumulator->values[BLACK],
        net->feature_transformer_weight[feature_index(accumulator->king_bucket[BLACK], BLACK, color, type, square)]);
}

void nnue_remove_feature(NnueAccumulator *accumulator, const Color color, const Piece type, const int square)
{
    const NnueNetwork *const net = accumulator->net;
    sub_column(
        accumulator->values[WHITE],
        net->feature_transformer_weight[feature_index(accumulator->king_bucket[WHITE], WHITE, color, type, square)]);
    sub_column(
        accumulator->values[BLACK],
        net->feature_transformer_weight[feature_index(accumulator->king_bucket[BLACK], BLACK, color, type, square)]);
}

void nnue_move_feature(NnueAccumulator *accumulator, const Color color, const Piece type, const int from, const int to)
{
    const NnueNetwork *const net = accumulator->net;
    for (int perspective = WHITE; perspective <= BLACK; perspective++)
    {
        const int bucket = accumulator->king_bucket[perspective];
        sub_column(accumulator->values[perspective],
                   net->feature_transformer_weight[feature_index(bucket, (Color)perspective, color, type, from)]);
        add_column(accumulator->values[perspective],
                   net->feature_transformer_weight[feature_index(bucket, (Color)perspective, color, type, to)]);
    }
}

/**
 * @brief Rebuild a single perspective's half (when that side's king bucket changes, and from refresh()).
 *
 * With a bound refresh cache (accumulator->cache, one per search thread): start from the cached accumulator
 * for this (perspective, bucket) and apply only the piece diffs versus the board it was built from — cost
 * proportional to pieces changed, not 32. An entry built for a different net (or never built: NULL) is a
 * miss and rebuilds from the bias, which replaces the old global net-generation guard. Without a cache the
 * half is rebuilt from scratch directly into the accumulator.
 */
void nnue_refresh_perspective(NnueAccumulator *accumulator, const Position *position, const Color perspective)
{
    const NnueNetwork *const net = accumulator->net;
    const int bucket = king_bucket(relative_king_square(perspective, position_king_sq(position, perspective)));
    accumulator->king_bucket[perspective] = bucket;

    if (accumulator->cache == NULL)
    {
        // No cache (standalone evals, verification walks): plain full rebuild of this half.
        memcpy(accumulator->values[perspective], net->feature_transformer_bias,
               sizeof(accumulator->values[perspective]));
        for (int color = WHITE; color <= BLACK; color++)
        {
            for (int type = PAWN; type <= KING; type++)
            {
                Bitboard pieces = position->colors[color] & position->pieces[type];
                while (pieces)
                {
                    const int square = pop_lsb(&pieces);
                    add_column(accumulator->values[perspective],
                               net->feature_transformer_weight[feature_index(bucket, perspective, (Color)color,
                                                                             (Piece)type, square)]);
                }
            }
        }
        return;
    }

    NnueRefreshCacheEntry *const cache = &accumulator->cache->entries[perspective][bucket];
    if (cache->net != net)
    {
        memcpy(cache->values, net->feature_transformer_bias, sizeof(cache->values));
        cache->colors[0] = cache->colors[1] = 0;
        for (int type = PAWN; type <= KING; type++)
        {
            cache->pieces[type] = 0;
        }
        cache->net = net;
    }

    for (int color = WHITE; color <= BLACK; color++)
    {
        for (int type = PAWN; type <= KING; type++)
        {
            const Bitboard current = position->colors[color] & position->pieces[type];
            const Bitboard cached  = cache->colors[color] & cache->pieces[type];
            Bitboard       added   = current & ~cached;
            Bitboard       removed = cached & ~current;
            while (added)
            {
                const int square = pop_lsb(&added);
                add_column(cache->values, net->feature_transformer_weight[feature_index(
                                              bucket, perspective, (Color)color, (Piece)type, square)]);
            }
            while (removed)
            {
                const int square = pop_lsb(&removed);
                sub_column(cache->values, net->feature_transformer_weight[feature_index(
                                              bucket, perspective, (Color)color, (Piece)type, square)]);
            }
        }
    }
    cache->colors[WHITE] = position->colors[WHITE];
    cache->colors[BLACK] = position->colors[BLACK];
    for (int type = PAWN; type <= KING; type++)
    {
        cache->pieces[type] = position->pieces[type];
    }
    memcpy(accumulator->values[perspective], cache->values, sizeof(cache->values));
}

void nnue_update_king_bucket(NnueAccumulator *accumulator, const Position *position, const Color side)
{
    const int new_bucket = king_bucket(relative_king_square(side, position_king_sq(position, side)));
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
/**
 * @brief SCReLU dot for one perspective: sum over i of clamp(acc[i],0,QA)^2 * weight[i], accumulated in int64.
 *
 * Bit-identical to the scalar reference: each term is (int64)(clamped*weight)*clamped (the middle product is
 * exact in int32 — clamped<=255, |weight|<=32767 => <=8.35M), and integer addition is associative so the
 * vector summation order does not matter. QA=255 fits int16, so the clamp is a plain int16 min/max.
 */
static inline int64_t screlu_dot(const int16_t *acc, const int16_t *weight)
{
    const __m256i zero = _mm256_setzero_si256();
    const __m256i qa   = _mm256_set1_epi16((int16_t)QUANT_ACCUMULATOR);
    __m256i       sum  = _mm256_setzero_si256(); // 4 int64 partial sums

    for (int i = 0; i < HIDDEN_SIZE; i += 16)
    {
        const __m256i a = _mm256_loadu_si256((const __m256i *)(acc + i));    // 16 int16 accumulator values
        const __m256i w = _mm256_loadu_si256((const __m256i *)(weight + i)); // 16 int16 output weights
        const __m256i c = _mm256_min_epi16(_mm256_max_epi16(a, zero), qa);   // clamp to [0,255] (16 int16)

        for (int half = 0; half < 2; half++)
        {
            const __m128i c128 = half ? _mm256_extracti128_si256(c, 1) : _mm256_castsi256_si128(c);
            const __m128i w128 = half ? _mm256_extracti128_si256(w, 1) : _mm256_castsi256_si128(w);
            const __m256i c32  = _mm256_cvtepi16_epi32(c128);  // 8 int32 clamped, [0,255]
            const __m256i w32  = _mm256_cvtepi16_epi32(w128);  // 8 int32 weights
            const __m256i p32  = _mm256_mullo_epi32(c32, w32); // clamped*weight, exact in int32
            // term = clamped * (clamped*weight) as int64. mul_epi32 reads the low 32 bits of each 64-bit lane
            // as a SIGNED int32, so even lanes come from (c32,p32) directly and odd lanes after a 32-bit shift.
            const __m256i even = _mm256_mul_epi32(c32, p32);
            const __m256i odd  = _mm256_mul_epi32(_mm256_srli_epi64(c32, 32), _mm256_srli_epi64(p32, 32));
            sum                = _mm256_add_epi64(sum, _mm256_add_epi64(even, odd));
        }
    }
    int64_t lanes[4];
    _mm256_storeu_si256((__m256i *)lanes, sum);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3];
}
#endif

int nnue_evaluate(const NnueAccumulator *accumulator, const Color stm)
{
    const NnueNetwork *const net      = accumulator->net;
    const int16_t *const     own      = accumulator->values[stm];
    const int16_t *const     opponent = accumulator->values[enemy_of(stm)];
#if defined(__AVX2__)
    int64_t accumulated = screlu_dot(own, net->output_weight) + screlu_dot(opponent, net->output_weight + HIDDEN_SIZE);
#else
    int64_t accumulated = 0;
    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        const int32_t own_clamped = clamp_accumulator(own[i]);
        accumulated += (int64_t)(own_clamped * net->output_weight[i]) * own_clamped;
        const int32_t opponent_clamped = clamp_accumulator(opponent[i]);
        accumulated += (int64_t)(opponent_clamped * net->output_weight[HIDDEN_SIZE + i]) * opponent_clamped;
    }
#endif
    accumulated /= QUANT_ACCUMULATOR;
    accumulated += net->output_bias;
    return (int)(accumulated * EVALUATION_SCALE / (QUANT_ACCUMULATOR * QUANT_OUTPUT));
}

int nnue_evaluate_position(const Position *position)
{
    NnueAccumulator accumulator;
    accumulator.net   = position->accumulator.net;
    accumulator.cache = NULL; // standalone eval: full rebuild, no per-thread cache
    nnue_refresh(&accumulator, position);
    return nnue_evaluate(&accumulator, position->color_to_move);
}

// ---- loading -----------------------------------------------------------------------------------------
const NnueNetwork *nnue_load(const char *path)
{
    FILE *const file = fopen(path, "rb");
    if (!file)
    {
        return NULL;
    }
    char magic[8];
    bool is_read_ok = fread(magic, 1, 8, file) == 8;
    if (!is_read_ok || memcmp(magic, NNUE_MAGIC, 8) != 0)
    {
        fprintf(stderr, "nnue: bad magic in %s\n", path);
        fclose(file);
        return NULL;
    }
    // The net is returned only on a fully-successful read, so a truncated file can never leave a caller
    // holding garbage weights.
    NnueNetwork *const loaded = malloc(sizeof(NnueNetwork));
    if (loaded == NULL)
    {
        fclose(file);
        fprintf(stderr, "nnue: out of memory loading %s\n", path);
        return NULL;
    }
    is_read_ok = is_read_ok && fread(loaded->feature_transformer_weight, 1, sizeof(loaded->feature_transformer_weight),
                                     file) == sizeof(loaded->feature_transformer_weight);
    is_read_ok = is_read_ok && fread(loaded->feature_transformer_bias, 1, sizeof(loaded->feature_transformer_bias),
                                     file) == sizeof(loaded->feature_transformer_bias);
    is_read_ok = is_read_ok &&
                 fread(loaded->output_weight, 1, sizeof(loaded->output_weight), file) == sizeof(loaded->output_weight);
    is_read_ok =
        is_read_ok && fread(&loaded->output_bias, 1, sizeof(loaded->output_bias), file) == sizeof(loaded->output_bias);
    fclose(file);
    if (!is_read_ok)
    {
        free(loaded);
        fprintf(stderr, "nnue: truncated file %s\n", path); // the caller's previous net (if any) stays intact
        return NULL;
    }
    return loaded;
}

void nnue_free(const NnueNetwork *net)
{
    free((void *)net);
}

// ---- verification helpers ----------------------------------------------------------------------------
int nnue_eval_fens_from_stdin(const char *net_path)
{
    const NnueNetwork *const net = nnue_load(net_path);
    if (net == NULL)
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
        position_init(&position, net);
        if (!position_set_fen(&position, line))
        {
            printf("0\n"); // malformed FEN: emit a placeholder so output stays line-aligned with the input
            continue;
        }
        printf("%d\n", nnue_evaluate_position(&position));
    }
    nnue_free(net);
    return 0;
}

/** @brief Tallies for one accumulator self-check walk (passed down the recursion, no file-scope state). */
typedef struct SelfCheckTally
{
    uint64_t nodes;      ///< positions visited
    uint64_t mismatches; ///< accumulator lanes that differed from a full refresh
    int      maxdiff;    ///< largest per-lane absolute difference seen
} SelfCheckTally;

/** @brief Recurse to @p depth comparing the incrementally-maintained accumulator against a full refresh. */
static void self_check_walk(const Position *position, const int depth, SelfCheckTally *tally)
{
    NnueAccumulator fresh;
    fresh.net   = position->accumulator.net;
    fresh.cache = NULL; // the reference side is always a full rebuild
    nnue_refresh(&fresh, position);
    for (int perspective = 0; perspective < 2; perspective++)
    {
        for (int i = 0; i < NNUE_HIDDEN; i++)
        {
            const int diff = abs(position->accumulator.values[perspective][i] - fresh.values[perspective][i]);
            if (diff)
            {
                tally->mismatches++;
            }
            if (diff > tally->maxdiff)
            {
                tally->maxdiff = diff;
            }
        }
    }
    tally->nodes++;
    if (depth == 0)
    {
        return;
    }
    Move moves[MAX_MOVES];
    generate_legal(position, moves, false);
    for (int index = 0; moves[index] != MOVE_NONE; index++)
    {
        Position child = *position;
        position_make_move(&child, moves[index]);
        self_check_walk(&child, depth - 1, tally);
    }
}

int nnue_run_self_check(const char *net_path)
{
    const NnueNetwork *const net = nnue_load(net_path);
    if (net == NULL)
    {
        fprintf(stderr, "nnue: failed to load %s\n", net_path);
        return 1;
    }
    // Bind a refresh cache into the walk's root so the finny diff-rebuild path is exercised, exactly as it
    // is in a real search thread; the walk's copy-make children inherit it.
    NnueRefreshCache refresh_cache;
    memset(&refresh_cache, 0, sizeof refresh_cache);
    const char *const fens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    };
    const size_t fen_count   = sizeof(fens) / sizeof(fens[0]);
    uint64_t     total_nodes = 0, total_mismatches = 0;
    int          overall_maxdiff = 0;
    double       total_secs      = 0.0;
    size_t       passed          = 0;
    for (size_t fen_index = 0; fen_index < fen_count; fen_index++)
    {
        Position position;
        position_init(&position, net);
        position.accumulator.cache = &refresh_cache;
        position_set_fen(&position, fens[fen_index]);
        SelfCheckTally tally    = {0};
        const int64_t  start_ms = platform_now_ms();
        self_check_walk(&position, 4, &tally);
        const double secs = (platform_now_ms() - start_ms) / 1000.0;
        total_nodes += tally.nodes;
        total_mismatches += tally.mismatches;
        total_secs += secs;
        passed += tally.mismatches == 0;
        if (tally.maxdiff > overall_maxdiff)
        {
            overall_maxdiff = tally.maxdiff;
        }
        char aux[32];
        snprintf(aux, sizeof aux, "max|diff| %d  %3llu mism", tally.maxdiff, (unsigned long long)tally.mismatches);
        test_result_columns(tally.mismatches == 0, "nnue", NULL, (int64_t)tally.nodes, aux, secs,
                            secs > 0.0 ? tally.nodes / secs / 1e6 : 0.0, fens[fen_index]);
    }
    char detail[16], aux[32];
    snprintf(detail, sizeof detail, "%zu/%zu", passed, fen_count);
    snprintf(aux, sizeof aux, "max|diff| %d  %3llu mism", overall_maxdiff, (unsigned long long)total_mismatches);
    test_result_columns(total_mismatches == 0, "nnue", detail, (int64_t)total_nodes, aux, total_secs,
                        total_secs > 0.0 ? total_nodes / total_secs / 1e6 : 0.0, "(incremental == refresh)");
    nnue_free(net);
    return total_mismatches ? 1 : 0;
}
