#include "tt.h"
#include <atomic>
#include <bit>

TranspositionTable TT;

namespace
{
// Pack the entry payload into one u64: move(16) | score(16) | eval(16) | depth(8) | bound(2) | gen(6).
inline uint64_t pack(uint16_t move, int16_t score, int16_t eval, uint8_t depth, uint8_t bound, uint8_t gen)
{
    return uint64_t(move) | (uint64_t(uint16_t(score)) << 16) | (uint64_t(uint16_t(eval)) << 32) |
           (uint64_t(depth) << 48) | (uint64_t(bound & 3) << 56) | (uint64_t(gen & 63) << 58);
}
} // namespace

void TranspositionTable::resize(size_t megabytes)
{
    size_t bytes      = megabytes * 1024 * 1024;
    size_t slot_count = bytes / sizeof(Slot);
    if (slot_count < 1024)
    {
        slot_count = 1024;
    }
    slot_count = std::bit_floor(slot_count); // power of two so index is a mask
    table.assign(slot_count, Slot{});
    mask       = slot_count - 1;
    generation = 0;
}

void TranspositionTable::clear()
{
    std::fill(table.begin(), table.end(), Slot{});
    generation = 0;
}

bool TranspositionTable::probe(uint64_t key, TTEntry &out)
{
    Slot    &slot    = table[key & mask];
    uint64_t data    = std::atomic_ref<uint64_t>(slot.data).load(std::memory_order_relaxed);
    uint64_t xor_key = std::atomic_ref<uint64_t>(slot.key).load(std::memory_order_relaxed);
    if ((xor_key ^ data) != key || data == 0)
    {
        return false; // miss, empty, or torn read
    }
    out.key   = key;
    out.move  = uint16_t(data & 0xFFFF);
    out.score = int16_t(uint16_t(data >> 16));
    out.eval  = int16_t(uint16_t(data >> 32));
    out.depth = int16_t(uint8_t(data >> 48));
    out.bound = uint8_t((data >> 56) & 3);
    out.gen   = uint8_t((data >> 58) & 63);
    return out.bound != BOUND_NONE;
}

void TranspositionTable::store(uint64_t key, int score, int eval, int depth, Bound bound, Move move, int ply)
{
    Slot    &slot               = table[key & mask];
    uint64_t current_data       = std::atomic_ref<uint64_t>(slot.data).load(std::memory_order_relaxed);
    uint64_t current_key        = std::atomic_ref<uint64_t>(slot.key).load(std::memory_order_relaxed);
    bool     same_key           = (current_data != 0) && ((current_key ^ current_data) == key);
    uint8_t  current_bound      = uint8_t((current_data >> 56) & 3);
    uint8_t  current_depth      = uint8_t(current_data >> 48);
    uint8_t  current_generation = uint8_t((current_data >> 58) & 63);

    // Preserve a TT move if the new store lacks one (a fail-low often has no best move).
    if (same_key && move.is_none())
    {
        move = Move(uint16_t(current_data & 0xFFFF));
    }
    // Replace when: empty/torn, this position, from an older search, or a deeper/exact result.
    if (current_data == 0 || current_bound == BOUND_NONE || same_key || current_generation != generation ||
        depth + (bound == BOUND_EXACT ? 2 : 0) >= current_depth)
    {
        uint64_t data =
            pack(move.raw(), int16_t(score_to_tt(score, ply)), int16_t(eval), uint8_t(depth), bound, generation);
        std::atomic_ref<uint64_t>(slot.key).store(key ^ data, std::memory_order_relaxed);
        std::atomic_ref<uint64_t>(slot.data).store(data, std::memory_order_relaxed);
    }
}

int TranspositionTable::hashfull()
{
    int used   = 0;
    int sample = table.size() < 1000 ? int(table.size()) : 1000;
    for (int i = 0; i < sample; i++)
    {
        uint64_t data = table[i].data;
        if (data != 0 && uint8_t((data >> 58) & 63) == generation)
        {
            used++;
        }
    }
    return used * 1000 / sample;
}
