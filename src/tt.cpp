#include "tt.h"
#include <atomic>
#include <bit>

TranspositionTable TT;

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

bool TranspositionTable::probe(uint64_t key, TTData &out) const
{
    const Slot &slot    = table[key & mask];
    uint64_t    data    = std::atomic_ref<const uint64_t>(slot.data).load(std::memory_order_relaxed);
    uint64_t    xor_key = std::atomic_ref<const uint64_t>(slot.key).load(std::memory_order_relaxed);
    if ((xor_key ^ data) != key || data == 0)
    {
        return false; // miss, empty, or torn read
    }
    out = std::bit_cast<TTData>(data); // one 64-bit copy; fields decode lazily at the use sites
    return out.bound != BOUND_NONE;
}

void TranspositionTable::store(uint64_t key, int score, int eval, int depth, Bound bound, Move move, int ply)
{
    Slot    &slot         = table[key & mask];
    uint64_t current_data = std::atomic_ref<uint64_t>(slot.data).load(std::memory_order_relaxed);
    uint64_t current_key  = std::atomic_ref<uint64_t>(slot.key).load(std::memory_order_relaxed);
    bool     same_key     = (current_data != 0) && ((current_key ^ current_data) == key);
    TTData   current      = std::bit_cast<TTData>(current_data);

    // Preserve a TT move if the new store lacks one (a fail-low often has no best move).
    if (same_key && move.is_none())
    {
        move = Move(uint16_t(current.move));
    }
    // Replace when: empty/torn, this position, from an older search, or a deeper/exact result.
    if (current_data == 0 || current.bound == BOUND_NONE || same_key || current.gen != generation ||
        uint64_t(depth + (bound == BOUND_EXACT ? 2 : 0)) >= current.depth)
    {
        TTData entry{};
        entry.move  = move.raw();
        entry.score = score_to_tt(score, ply);
        entry.eval  = eval;
        entry.depth = uint64_t(depth);
        entry.bound = bound;
        entry.gen   = generation;

        uint64_t data = std::bit_cast<uint64_t>(entry);
        std::atomic_ref<uint64_t>(slot.key).store(key ^ data, std::memory_order_relaxed);
        std::atomic_ref<uint64_t>(slot.data).store(data, std::memory_order_relaxed);
    }
}

int TranspositionTable::hashfull() const
{
    int used   = 0;
    int sample = table.size() < 1000 ? int(table.size()) : 1000;
    for (int i = 0; i < sample; i++)
    {
        uint64_t data = table[i].data;
        if (data != 0 && std::bit_cast<TTData>(data).gen == generation)
        {
            used++;
        }
    }
    return used * 1000 / sample;
}
