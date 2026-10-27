// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief NNUE internals: king-bucketed feature indexing, incremental accumulator maintenance, and the integer forward.
 */
#include "nnue.h"
#include "crc64.h"
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
    HIDDEN_SIZE       = NNUE_HIDDEN, // 1024
    BASE_FEATURES     = 768,         // per-king-bucket block (2 colors x 6 types x 64 squares)
    NUM_KING_BUCKETS  = 8,           // 4 files (a-d, after mirroring) x 2 board-halves, on the perspective king
    INPUT_FEATURES    = NUM_KING_BUCKETS * BASE_FEATURES, // 6144 feature-transformer rows
    QUANT_ACCUMULATOR = 255,                              // QA
    QUANT_OUTPUT      = 64,                               // QB
    EVALUATION_SCALE  = 400
};

/**
 * @brief Whether a perspective whose king sits on files e-h is mirrored horizontally before indexing.
 *
 * Always on in Zenith — the shape the shared evaluator standardised on is Zenith's, mirroring included
 * (+14.3 Elo, SPRT-gated here). It is a named constant rather than a literal because the container's
 * format version and its mirroring flag are both *derived* from it, exactly as the C++ port derives them
 * from its build knob: turning mirroring off would make this a different format, not the same format with
 * a flag cleared, because the king-bucket map's contents change with it.
 */
enum
{
    IS_KING_MIRRORING_ENABLED = 1
};

/** @brief The shared Reckless net container, `RNNUE1` — one format for every engine in the organisation.
 *
 * The retired `ZNNUE5`/`ZNNUE4`/`ZNNUE3` formats carried an 8-byte magic and nothing else, so a net for a
 * different architecture was told apart only by its file size. This header describes itself: every
 * architecture field is checked against this build's compile-time constants, the king-bucket map's
 * *contents* are checked by digest (two maps of the same cardinality used to be indistinguishable, which
 * loaded cleanly and evaluated from the wrong weight bank), and three CRCs make a corrupt or truncated
 * file refuse rather than play. See the `reckless-nnue` skill for the specification. */
enum
{
    NET_FORMAT_VERSION      = IS_KING_MIRRORING_ENABLED ? 2 : 1, ///< 2 = mirrored map + output heads
    NET_HEADER_BYTES        = 64,                                ///< lets a later version extend the header
    NET_FLAG_KING_MIRRORING = 0x01,                              ///< flags bit 0: horizontal king mirroring
    NET_FLAGS               = IS_KING_MIRRORING_ENABLED ? NET_FLAG_KING_MIRRORING : 0,
    NET_ACTIVATION_SCRELU   = 0, ///< activation 0: clamp(x, 0, QA), which the output dot then squares
    /// The payload's four sections, in the order they are stored: three int16 arrays then one int32 per
    /// output bucket, tightly packed with nothing after them. 12,617,760 bytes at this shape.
    NET_INT16_ELEMENTS = INPUT_FEATURES * HIDDEN_SIZE + HIDDEN_SIZE + NNUE_OUTPUT_BUCKETS * 2 * HIDDEN_SIZE,
    NET_PAYLOAD_BYTES  = NET_INT16_ELEMENTS * 2 + NNUE_OUTPUT_BUCKETS * 4
};

static const char NET_MAGIC[8] = {'R', 'N', 'N', 'U', 'E', '1', '\0', '\0'};

/** @brief The 64-byte self-describing header every RNNUE1 net carries, little-endian. Every field is
 *  naturally aligned at these offsets, so no packing attribute is needed — the static assertions below
 *  prove the layout rather than trusting it. */
typedef struct NetHeader
{
    char     magic[8];            ///< NET_MAGIC, at offset 0
    uint16_t format_version;      ///< NET_FORMAT_VERSION, at 8
    uint16_t header_bytes;        ///< NET_HEADER_BYTES, at 10
    uint16_t features_per_bucket; ///< BASE_FEATURES, at 12
    uint16_t king_buckets;        ///< NUM_KING_BUCKETS, at 14
    uint16_t hidden_size;         ///< HIDDEN_SIZE, at 16
    uint16_t output_buckets;      ///< NNUE_OUTPUT_BUCKETS, at 18
    uint8_t  flags;               ///< NET_FLAGS, at 20; every other bit zero
    uint8_t  activation;          ///< NET_ACTIVATION_SCRELU, at 21
    uint16_t qa;                  ///< QUANT_ACCUMULATOR, at 22 (read unsigned)
    uint16_t qb;                  ///< QUANT_OUTPUT, at 24 (read unsigned)
    uint16_t eval_scale;          ///< EVALUATION_SCALE, at 26 (read unsigned)
    uint8_t  reserved[4];         ///< zero, at 28 — refused otherwise; see nnue_parse
    uint64_t king_bucket_map_crc; ///< CRC-64/XZ of KING_BUCKET_MAP's 64 entries as uint8, at 32
    uint64_t payload_bytes;       ///< NET_PAYLOAD_BYTES, at 40
    uint64_t payload_crc;         ///< CRC-64/XZ over the payload, at 48
    uint64_t header_crc;          ///< CRC-64/XZ over this header's first 56 bytes, at 56
} NetHeader;

_Static_assert(sizeof(NetHeader) == NET_HEADER_BYTES, "the RNNUE1 header must be exactly 64 bytes");
_Static_assert(offsetof(NetHeader, flags) == 20, "flags must sit at offset 20");
_Static_assert(offsetof(NetHeader, king_bucket_map_crc) == 32, "king_bucket_map_crc must sit at offset 32");
_Static_assert(offsetof(NetHeader, payload_bytes) == 40, "payload_bytes must sit at offset 40");
_Static_assert(offsetof(NetHeader, header_crc) == 56, "header_crc must sit at offset 56");

/// Bytes of the header that header_crc covers: everything before header_crc itself.
#define NET_HEADER_CRC_BYTES offsetof(NetHeader, header_crc)

/** @brief The quantised network weights, loaded from a `.nnue` file (completes accumulator.h's forward decl). */
struct NnueNetwork
{
    int16_t feature_transformer_weight[INPUT_FEATURES][HIDDEN_SIZE]; ///< feature-major, king-bucketed
    int16_t feature_transformer_bias[HIDDEN_SIZE];                   ///< feature-transformer bias
    /// Output heads, one per material bucket ((piece_count - 2) / 4): own half [0,HIDDEN), opponent half
    /// [HIDDEN,2*HIDDEN).
    int16_t output_weight[NNUE_OUTPUT_BUCKETS][2 * HIDDEN_SIZE];
    int32_t output_bias[NNUE_OUTPUT_BUCKETS]; ///< output bias per material bucket
};

_Static_assert((int)NNUE_KING_BUCKETS == (int)NUM_KING_BUCKETS, "header/implementation bucket counts must agree");
_Static_assert(sizeof(((struct NnueNetwork *)0)->feature_transformer_weight) +
                       sizeof(((struct NnueNetwork *)0)->feature_transformer_bias) +
                       sizeof(((struct NnueNetwork *)0)->output_weight) +
                       sizeof(((struct NnueNetwork *)0)->output_bias) ==
                   NET_PAYLOAD_BYTES,
               "NET_PAYLOAD_BYTES must be the four weight arrays back to back");

/**
 * @brief King-input bucket per perspective-relative king square: 4 files (a-d) x 2 board-halves.
 *
 * `half * 4 + min(file, 7 - file)` — per *file*, because mirroring has already folded e-h onto d-a, so
 * every bucket trains on both wings. Held as the one array the evaluator itself indexes, and digested into
 * the header from that same array: the whole point of `king_bucket_map_crc` is that there is no second
 * representation of the map to drift out of step with this one. Must match king_bucket() in
 * trainer/features.py, which generates this table.
 */
static const uint8_t KING_BUCKET_MAP[64] = {
    0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0,
    4, 5, 6, 7, 7, 6, 5, 4, 4, 5, 6, 7, 7, 6, 5, 4, 4, 5, 6, 7, 7, 6, 5, 4, 4, 5, 6, 7, 7, 6, 5, 4,
};

/** @brief Whether a perspective mirrors horizontally: its own king is on files e-h. */
static inline bool king_mirror(const int relative_king_square)
{
    return IS_KING_MIRRORING_ENABLED && (relative_king_square & 7) >= 4;
}

/** @brief King-input bucket for a perspective-relative king square (0..63). */
static inline int king_bucket(const int relative_king_square)
{
    return KING_BUCKET_MAP[relative_king_square];
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
    const NnueNetwork *const net      = accumulator->net;
    const int                sq_white = accumulator->king_mirror[WHITE] ? square ^ 7 : square;
    const int                sq_black = accumulator->king_mirror[BLACK] ? square ^ 7 : square;
    add_column(
        accumulator->values[WHITE],
        net->feature_transformer_weight[feature_index(accumulator->king_bucket[WHITE], WHITE, color, type, sq_white)]);
    add_column(
        accumulator->values[BLACK],
        net->feature_transformer_weight[feature_index(accumulator->king_bucket[BLACK], BLACK, color, type, sq_black)]);
}

void nnue_remove_feature(NnueAccumulator *accumulator, const Color color, const Piece type, const int square)
{
    const NnueNetwork *const net      = accumulator->net;
    const int                sq_white = accumulator->king_mirror[WHITE] ? square ^ 7 : square;
    const int                sq_black = accumulator->king_mirror[BLACK] ? square ^ 7 : square;
    sub_column(
        accumulator->values[WHITE],
        net->feature_transformer_weight[feature_index(accumulator->king_bucket[WHITE], WHITE, color, type, sq_white)]);
    sub_column(
        accumulator->values[BLACK],
        net->feature_transformer_weight[feature_index(accumulator->king_bucket[BLACK], BLACK, color, type, sq_black)]);
}

void nnue_move_feature(NnueAccumulator *accumulator, const Color color, const Piece type, const int from, const int to)
{
    const NnueNetwork *const net = accumulator->net;
    for (int perspective = WHITE; perspective <= BLACK; perspective++)
    {
        const int bucket  = accumulator->king_bucket[perspective];
        const int sq_from = accumulator->king_mirror[perspective] ? from ^ 7 : from;
        const int sq_to   = accumulator->king_mirror[perspective] ? to ^ 7 : to;
        sub_column(accumulator->values[perspective],
                   net->feature_transformer_weight[feature_index(bucket, (Color)perspective, color, type, sq_from)]);
        add_column(accumulator->values[perspective],
                   net->feature_transformer_weight[feature_index(bucket, (Color)perspective, color, type, sq_to)]);
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
    const NnueNetwork *const net           = accumulator->net;
    const int                relative_king = relative_king_square(perspective, position->king_location[perspective]);
    const bool               mirror        = king_mirror(relative_king);
    const int                bucket        = king_bucket(relative_king);
    accumulator->king_bucket[perspective]  = bucket;
    accumulator->king_mirror[perspective]  = (uint8_t)mirror;

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
                    const int raw    = pop_lsb(&pieces);
                    const int square = mirror ? raw ^ 7 : raw;
                    add_column(accumulator->values[perspective],
                               net->feature_transformer_weight[feature_index(bucket, perspective, (Color)color,
                                                                             (Piece)type, square)]);
                }
            }
        }
        return;
    }

    NnueRefreshCacheEntry *const cache = &accumulator->cache->entries[perspective][mirror ? 1 : 0][bucket];
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
                const int square = mirror ? pop_lsb(&added) ^ 7 : pop_lsb(&added);
                add_column(cache->values, net->feature_transformer_weight[feature_index(
                                              bucket, perspective, (Color)color, (Piece)type, square)]);
            }
            while (removed)
            {
                const int square = mirror ? pop_lsb(&removed) ^ 7 : pop_lsb(&removed);
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
    const int relative_king = relative_king_square(side, position->king_location[side]);
    // A refresh is needed when the bucket OR the mirror state changes — crossing the d/e file boundary can
    // flip the mirror while landing in the same bucket index, which rebases every feature in the perspective.
    if (king_bucket(relative_king) != accumulator->king_bucket[side] ||
        (uint8_t)king_mirror(relative_king) != accumulator->king_mirror[side])
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

int nnue_evaluate(const NnueAccumulator *accumulator, const Color stm, const int output_bucket)
{
    const NnueNetwork *const net      = accumulator->net;
    const int16_t *const     weights  = net->output_weight[output_bucket];
    const int16_t *const     own      = accumulator->values[stm];
    const int16_t *const     opponent = accumulator->values[enemy_of(stm)];
#if defined(__AVX2__)
    int64_t accumulated = screlu_dot(own, weights) + screlu_dot(opponent, weights + HIDDEN_SIZE);
#else
    int64_t accumulated = 0;
    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        const int32_t own_clamped = clamp_accumulator(own[i]);
        accumulated += (int64_t)(own_clamped * weights[i]) * own_clamped;
        const int32_t opponent_clamped = clamp_accumulator(opponent[i]);
        accumulated += (int64_t)(opponent_clamped * weights[HIDDEN_SIZE + i]) * opponent_clamped;
    }
#endif
    accumulated /= QUANT_ACCUMULATOR;
    accumulated += net->output_bias[output_bucket];
    return (int)(accumulated * EVALUATION_SCALE / (QUANT_ACCUMULATOR * QUANT_OUTPUT));
}

int nnue_evaluate_position(const Position *position)
{
    NnueAccumulator accumulator;
    accumulator.net   = position->accumulator.net;
    accumulator.cache = NULL; // standalone eval: full rebuild, no per-thread cache
    nnue_refresh(&accumulator, position);
    return nnue_evaluate(&accumulator, position->color_to_move, nnue_output_bucket(position));
}

// ---- loading -----------------------------------------------------------------------------------------
/**
 * @brief Report a refused net and return NULL, so a caller can `return refuse_net(...)`.
 *
 * Every refusal goes through here, which puts the one rule that matters in one place: **a refused net is
 * not loaded, and nothing falls back to another net.** A stock engine that silently fell back to its
 * embedded net once reported a plausible Elo for a net that never played a move — the match looked fine
 * and the number meant nothing.
 *
 * To stderr, which is what a match harness keeps. The UCI layer adds an `info string` line on stdout for a
 * GUI to read; it is not done here because `nnueeval` writes one eval per line to stdout and a diagnostic
 * mixed into that stream would corrupt the 0 cp verification gate's input.
 * @param source_name path or label of the net. @param reason why it was refused, naming the field and both
 * values where there is a field to name. @return NULL, always.
 */
static const NnueNetwork *refuse_net(const char *source_name, const char *reason)
{
    fprintf(stderr, "nnue: refusing net '%s': %s\n", source_name, reason);
    return NULL;
}

/** @brief CRC-64/XZ of KING_BUCKET_MAP's 64 entries — the digest the header carries so that two maps of
 *  the same cardinality can no longer be mistaken for one another. @return the digest. */
static uint64_t king_bucket_map_crc(void)
{
    return crc64_checksum(KING_BUCKET_MAP, sizeof KING_BUCKET_MAP);
}

/** @brief The architecture this build was compiled for, as a header, with the two computed CRCs left zero
 *  (only something holding the payload can fill them). @return the expected header. */
static NetHeader expected_net_header(void)
{
    NetHeader header;
    memset(&header, 0, sizeof header);
    memcpy(header.magic, NET_MAGIC, sizeof header.magic);
    header.format_version      = NET_FORMAT_VERSION;
    header.header_bytes        = NET_HEADER_BYTES;
    header.features_per_bucket = BASE_FEATURES;
    header.king_buckets        = NUM_KING_BUCKETS;
    header.hidden_size         = HIDDEN_SIZE;
    header.output_buckets      = NNUE_OUTPUT_BUCKETS;
    header.flags               = NET_FLAGS;
    header.activation          = NET_ACTIVATION_SCRELU;
    header.qa                  = QUANT_ACCUMULATOR;
    header.qb                  = QUANT_OUTPUT;
    header.eval_scale          = EVALUATION_SCALE;
    header.king_bucket_map_crc = king_bucket_map_crc();
    header.payload_bytes       = NET_PAYLOAD_BYTES;
    return header;
}

/**
 * @brief Parse a complete RNNUE1 image from memory, committing only on a fully-successful parse so a
 * refused image can never leave a caller holding garbage weights.
 *
 * The two checks the shared container defines, in this order:
 *   1. **the header** — 64 bytes: the magic, then `header_crc` over its own first 56 bytes, then every
 *      architecture field against this build's compile-time constants, then the king-bucket map's digest
 *      against this build's own map;
 *   2. **the payload** — its length must be *exactly* what the header claims and what this architecture
 *      expects, with nothing appended, and then its checksum must match.
 *
 * @p source_name labels the messages. @return a heap-allocated net, or NULL on any refusal.
 */
static const NnueNetwork *nnue_parse(const unsigned char *bytes, const size_t size, const char *source_name)
{
    char message[256];
    if (size < sizeof(NetHeader))
    {
        snprintf(message, sizeof message, "it is %zu bytes, too short for the %zu-byte RNNUE1 header", size,
                 sizeof(NetHeader));
        return refuse_net(source_name, message);
    }
    if (memcmp(bytes, NET_MAGIC, sizeof NET_MAGIC) != 0)
    {
        return refuse_net(source_name, "it carries no 'RNNUE1' magic, so it is not a net in the shared container — "
                                       "a net in a retired ZNNUE format must be re-stamped (tools/stamp_net.py)");
    }

    NetHeader header;
    memcpy(&header, bytes, sizeof header);
    const uint64_t header_crc = crc64_checksum(bytes, NET_HEADER_CRC_BYTES);
    if (header.header_crc != header_crc)
    {
        snprintf(message, sizeof message,
                 "its header is corrupt: header_crc is %#018llx, the bytes checksum to %#018llx",
                 (unsigned long long)header.header_crc, (unsigned long long)header_crc);
        return refuse_net(source_name, message);
    }

    const NetHeader expected = expected_net_header();

    const struct
    {
        const char *name;
        uint64_t    in_file;
        uint64_t    in_engine;
    } checks[] = {
        {"format_version", header.format_version, expected.format_version},
        {"header_bytes", header.header_bytes, expected.header_bytes},
        {"features_per_bucket", header.features_per_bucket, expected.features_per_bucket},
        {"king_buckets", header.king_buckets, expected.king_buckets},
        {"hidden_size", header.hidden_size, expected.hidden_size},
        {"output_buckets", header.output_buckets, expected.output_buckets},
        {"flags", header.flags, expected.flags},
        {"activation", header.activation, expected.activation},
        {"qa", header.qa, expected.qa},
        {"qb", header.qb, expected.qb},
        {"eval_scale", header.eval_scale, expected.eval_scale},
        {"king_bucket_map_crc", header.king_bucket_map_crc, expected.king_bucket_map_crc},
        {"payload_bytes", header.payload_bytes, expected.payload_bytes},
    };

    for (size_t index = 0; index < sizeof checks / sizeof checks[0]; index++)
    {
        if (checks[index].in_file != checks[index].in_engine)
        {
            snprintf(message, sizeof message, "%s is %#llx in the file, %#llx in this engine", checks[index].name,
                     (unsigned long long)checks[index].in_file, (unsigned long long)checks[index].in_engine);
            return refuse_net(source_name, message);
        }
    }

    // `reserved` is refused unless every byte is zero. Those four bytes are the only room a later format
    // version has to add a field without moving anything, and that only works if a reader of *this* version
    // refuses a net that has already spent them — otherwise a later net loads here with one field unread.
    for (size_t index = 0; index < sizeof header.reserved; index++)
    {
        if (header.reserved[index] != 0)
        {
            snprintf(message, sizeof message,
                     "its reserved bytes are 0x%02x 0x%02x 0x%02x 0x%02x, and every one must be zero",
                     (unsigned)header.reserved[0], (unsigned)header.reserved[1], (unsigned)header.reserved[2],
                     (unsigned)header.reserved[3]);
            return refuse_net(source_name, message);
        }
    }

    const size_t payload_bytes = size - sizeof(NetHeader);
    if (payload_bytes != (size_t)NET_PAYLOAD_BYTES)
    {
        snprintf(message, sizeof message, "it carries %zu payload bytes after the header, not the %d claimed",
                 payload_bytes, NET_PAYLOAD_BYTES);
        return refuse_net(source_name, message);
    }
    const unsigned char *payload     = bytes + sizeof(NetHeader);
    const uint64_t       payload_crc = crc64_checksum(payload, payload_bytes);
    if (header.payload_crc != payload_crc)
    {
        snprintf(message, sizeof message,
                 "its payload is corrupt: payload_crc is %#018llx, the bytes checksum to %#018llx",
                 (unsigned long long)header.payload_crc, (unsigned long long)payload_crc);
        return refuse_net(source_name, message);
    }

    NnueNetwork *const loaded = malloc(sizeof(NnueNetwork));
    if (loaded == NULL)
    {
        return refuse_net(source_name, "this process is out of memory");
    }
    // Every check above ran before a single weight was written, so nothing partial can escape. Copy in the
    // payload's order: feature weights, feature bias, output weights, output bias.
    memcpy(loaded->feature_transformer_weight, payload, sizeof loaded->feature_transformer_weight);
    payload += sizeof loaded->feature_transformer_weight;
    memcpy(loaded->feature_transformer_bias, payload, sizeof loaded->feature_transformer_bias);
    payload += sizeof loaded->feature_transformer_bias;
    memcpy(loaded->output_weight, payload, sizeof loaded->output_weight);
    payload += sizeof loaded->output_weight;
    memcpy(loaded->output_bias, payload, sizeof loaded->output_bias);
    return loaded;
}

const NnueNetwork *nnue_load(const char *path)
{
    FILE *const file = fopen(path, "rb");
    if (!file)
    {
        return NULL;
    }
    // Read the whole file into a temporary buffer and parse from memory — one code path shared with the
    // embedded net. The buffer is a transient ~12.6MB; the parsed net is the only long-lived allocation.
    fseek(file, 0, SEEK_END);
    const long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    if (file_size <= 0)
    {
        fclose(file);
        return refuse_net(path, "it holds no bytes");
    }
    unsigned char *const bytes = malloc((size_t)file_size);
    if (bytes == NULL)
    {
        fclose(file);
        return refuse_net(path, "this process is out of memory");
    }
    const bool is_read_ok = fread(bytes, 1, (size_t)file_size, file) == (size_t)file_size;
    fclose(file);
    const NnueNetwork *const loaded =
        is_read_ok ? nnue_parse(bytes, (size_t)file_size, path) : refuse_net(path, "it opened but could not be read");
    free(bytes);
    return loaded;
}

const NnueNetwork *nnue_load_embedded(void)
{
    return nnue_parse(embedded_network_data, embedded_network_size, "<embedded net>");
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

enum
{
    SELF_CHECK_DEPTH = 4 ///< plies walked per position
};

/** @brief Tallies for one accumulator self-check walk (passed down the recursion, no file-scope state). */
typedef struct SelfCheckTally
{
    uint64_t nodes;      ///< positions visited
    uint64_t mismatches; ///< accumulator lanes that differed from a full refresh
    int      maxdiff;    ///< largest per-lane absolute difference seen
} SelfCheckTally;

/** @brief Compare @p position's incrementally-maintained accumulator against a full refresh, lane by lane. */
static void self_check_node(const Position *position, SelfCheckTally *tally)
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
}

/** @brief Recurse to @p depth applying self_check_node at every position reached. */
static void self_check_walk(const Position *position, const int depth, SelfCheckTally *tally)
{
    self_check_node(position, tally);
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

/**
 * @brief self_check_walk over @p position, refreshing the tty progress line once per root move.
 *
 * The root ply is expanded here rather than reporting from inside self_check_walk, for the same reason
 * run_legal_check does it: a progress path in the hot recursive frame costs real throughput.
 */
static void self_check_position(const Position *position, SelfCheckTally *tally, const char *case_detail,
                                const int64_t start_ms)
{
    self_check_node(position, tally);
    Move root_moves[MAX_MOVES];
    generate_legal(position, root_moves, false);
    for (int index = 0; root_moves[index] != MOVE_NONE; index++)
    {
        Position child = *position;
        position_make_move(&child, root_moves[index]);
        self_check_walk(&child, SELF_CHECK_DEPTH - 1, tally);
        test_progress("nnue", case_detail, (int64_t)tally->nodes, (double)(platform_now_ms() - start_ms) / 1000.0);
    }
}

int nnue_run_self_check(const char *net_path)
{
    // Before anything looks at a net: this build must reproduce the published CRC-64/XZ check value. Every
    // Reckless engine asserts the same one, and a CRC with the wrong reflection or a wrong table constant
    // still returns a plausible 64-bit number — so without this gate the first symptom would be each
    // engine refusing every net the others stamped. C17 cannot assert it at compile time (no constant-
    // evaluated loop), which is why it is the first thing `nnuecheck` reports.
    char crc_aux[40];
    snprintf(crc_aux, sizeof crc_aux, "%#018llx", (unsigned long long)CRC64_CHECK_VALUE);
    const bool is_crc_ok = crc64_is_check_value_reproduced();
    test_result_columns(is_crc_ok, "crc64", "check", 9, crc_aux, -1.0, -1.0,
                        "(CRC-64/XZ of the 9 bytes \"123456789\")");
    if (!is_crc_ok)
    {
        fprintf(stderr, "nnue: this build's CRC-64/XZ is wrong — it would refuse every shared net\n");
        return 1;
    }

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
    // Cheapest-first (see the same note in run_legal_check): the heavy position runs last so the table does not
    // stall right after its first line.
    const char *const fens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
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
        char case_detail[16];
        snprintf(case_detail, sizeof case_detail, "#%zu/%zu", fen_index + 1, fen_count);
        SelfCheckTally tally    = {0};
        const int64_t  start_ms = platform_now_ms();
        self_check_position(&position, &tally, case_detail, start_ms);
        const double secs = (platform_now_ms() - start_ms) / 1000.0;
        test_progress_clear();
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
