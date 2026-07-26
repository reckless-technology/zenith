// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief Transposition-table storage: allocation, lockless probe/store, and fill estimation.
 */
#include "tt.h"
#include <stdlib.h>

static int clamp_int(const int value, const int low, const int high)
{
    return value < low ? low : (value > high ? high : value);
}

/** @brief Largest power of two <= @p x (x > 0), so the table size becomes an index mask. */
static size_t bit_floor_size(const size_t x)
{
    return (size_t)1 << (63 - __builtin_clzll((uint64_t)x));
}

void tt_resize(TranspositionTable *tt, size_t megabytes)
{
    // Clamp to the advertised UCI range before the byte multiply, so a negative/huge `Hash` value (it reaches
    // here as a wrapped size_t via atoi) cannot overflow `bytes` and request an absurd allocation.
    if (megabytes < 1)
    {
        megabytes = 1;
    }
    if (megabytes > TT_MAX_MB)
    {
        megabytes = TT_MAX_MB;
    }
    const size_t bytes      = megabytes * 1024u * 1024u;
    size_t       slot_count = bytes / sizeof(TTSlot);
    if (slot_count < 1024)
    {
        slot_count = 1024;
    }
    slot_count = bit_floor_size(slot_count); // power of two so index is a mask

    TTSlot *fresh = calloc(slot_count, sizeof(TTSlot));
    while (fresh == NULL && slot_count > 1024)
    {
        slot_count /= 2; // back off on allocation failure rather than run with a NULL table
        fresh = calloc(slot_count, sizeof(TTSlot));
    }
    if (fresh == NULL)
    {
        return; // keep the existing table (never leave tt->table NULL for probe/store to dereference)
    }
    free(tt->table);
    tt->table      = fresh;
    tt->slot_count = slot_count;
    tt->mask       = slot_count - 1;
    tt->generation = 0;
}

void tt_clear(TranspositionTable *tt)
{
    memset((void *)tt->table, 0, tt->slot_count * sizeof(TTSlot));
    tt->generation = 0;
}

bool tt_probe(const TranspositionTable *tt, const uint64_t key, TTData *out)
{
    const TTSlot *const slot    = &tt->table[key & tt->mask];
    const uint64_t      data    = atomic_load_explicit(&slot->data, memory_order_relaxed);
    const uint64_t      xor_key = atomic_load_explicit(&slot->key, memory_order_relaxed);
    if ((xor_key ^ data) != key || data == 0)
    {
        return false; // miss, empty, or torn read
    }
    *out = u64_to_tt_data(data); // one 64-bit copy; fields decode lazily at the use sites
    return out->bound != BOUND_NONE;
}

void tt_store(TranspositionTable *tt, const uint64_t key, const int score, const int eval, const int depth,
              const Bound bound, Move move, const int ply)
{
    TTSlot *const  slot         = &tt->table[key & tt->mask];
    const uint64_t current_data = atomic_load_explicit(&slot->data, memory_order_relaxed);
    const uint64_t current_key  = atomic_load_explicit(&slot->key, memory_order_relaxed);
    const bool     is_same_key  = (current_data != 0) && ((current_key ^ current_data) == key);
    const TTData   current      = u64_to_tt_data(current_data);

    // Preserve a TT move if the new store lacks one (a fail-low often has no best move).
    if (is_same_key && move_is_none(move))
    {
        move = (Move)current.move;
    }
    // Replace when: empty/torn, this position, from an older search, or a deeper/exact result.
    // depth comparisons are SIGNED — a qsearch-style entry (depth <= 0) must lose to any real depth.
    if (current_data == 0 || current.bound == BOUND_NONE || is_same_key || current.gen != tt->generation ||
        depth + (bound == BOUND_EXACT ? 2 : 0) >= (int)current.depth)
    {
        TTData entry = {0};
        entry.move   = move;
        entry.score  = score_to_tt(score, ply);
        entry.eval   = eval;
        entry.depth  = clamp_int(depth, -128, 127); // signed :8 field — clamp, never wrap
        entry.bound  = bound;
        entry.gen    = tt->generation;

        const uint64_t data = tt_data_to_u64(entry);
        atomic_store_explicit(&slot->key, key ^ data, memory_order_relaxed);
        atomic_store_explicit(&slot->data, data, memory_order_relaxed);
    }
}

int tt_hashfull(const TranspositionTable *tt)
{
    int       used   = 0;
    const int sample = tt->slot_count < 1000 ? (int)tt->slot_count : 1000;
    for (int i = 0; i < sample; i++)
    {
        const uint64_t data = atomic_load_explicit(&tt->table[i].data, memory_order_relaxed);
        if (data != 0 && u64_to_tt_data(data).gen == tt->generation)
        {
            used++;
        }
    }
    return used * 1000 / sample;
}
