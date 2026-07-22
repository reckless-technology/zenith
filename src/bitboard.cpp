#include "bitboard.h"

Bitboard PawnAttacks[COLOR_NB][64];
Bitboard KnightAttacks[64];
Bitboard KingAttacks[64];
Bitboard BetweenBB[64][64];
Bitboard LineBB[64][64];

namespace
{

// Deterministic sparse PRNG for magic search (xorshift64*), seeded once.
struct PRNG
{
    uint64_t state;

    explicit PRNG(uint64_t seed) : state(seed)
    {
    }

    uint64_t next()
    {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        return state * 0x2545F4914F6CDD1DULL;
    }

    uint64_t sparse()
    {
        return next() & next() & next();
    } // few set bits -> good magic candidates
};

// Ray-walk sliding attacks (used to build masks and to fill the magic tables).
Bitboard sliding_attack(int square, Bitboard occupancy, const int deltas[4])
{
    Bitboard attacks = 0;
    for (int direction = 0; direction < 4; direction++)
    {
        int delta   = deltas[direction];
        int current = square;
        while (true)
        {
            int previousFile = file_of(current);
            int nextSquare   = current + delta;
            if (nextSquare < 0 || nextSquare >= 64)
            {
                break;
            }
            // reject rank/file wraps: any king-step move changes file by at most 1
            if (std::abs(file_of(nextSquare) - previousFile) > 1)
            {
                break;
            }
            attacks |= sq_bb(nextSquare);
            if (occupancy & sq_bb(nextSquare))
            {
                break; // blocker occupies this square; stop after including it
            }
            current = nextSquare;
        }
    }
    return attacks;
}

const int RookDirs[4]   = {NORTH, SOUTH, EAST, WEST};
const int BishopDirs[4] = {NE, NW, SE, SW};

struct Magic
{
    Bitboard  mask    = 0;
    Bitboard  magic   = 0;
    Bitboard *attacks = nullptr;
    unsigned  shift   = 0;

    unsigned index(Bitboard occupancy) const
    {
        return unsigned(((occupancy & mask) * magic) >> shift);
    }
};

Magic    RookMagics[64];
Magic    BishopMagics[64];
Bitboard RookTable[0x19000];  // 102400
Bitboard BishopTable[0x1480]; // 5248

void init_magics(bool rook, Bitboard *table, Magic magics[64], const int deltas[4])
{
    Bitboard  occupancy[4096], reference[4096];
    int       epoch[4096]  = {0};
    int       epochCounter = 0;
    Bitboard *attackBase   = table;

    for (int square = 0; square < 64; square++)
    {
        // Relevant-occupancy mask = empty-board rays minus the edges not on the piece's own rank/file.
        Bitboard edges         = ((RANK_1 | RANK_8) & ~rank_bb(square)) | ((FILE_A | FILE_H) & ~file_bb(square));
        Bitboard mask          = sliding_attack(square, 0, deltas) & ~edges;
        unsigned relevantBits  = popcount(mask);
        magics[square].mask    = mask;
        magics[square].shift   = 64 - relevantBits;
        magics[square].attacks = attackBase;

        // Enumerate every subset of mask (Carry-Rippler), recording its true attack set.
        Bitboard subset      = 0;
        int      subsetCount = 0;
        do
        {
            occupancy[subsetCount] = subset;
            reference[subsetCount] = sliding_attack(square, subset, deltas);
            subsetCount++;
            subset = (subset - mask) & mask;
        } while (subset);

        PRNG prng(0x9E3779B97F4A7C15ULL ^ (uint64_t(square) * 0xBF58476D1CE4E5B9ULL) ^ (rook ? 1 : 2));
        for (int subsetIndex = 0; subsetIndex < subsetCount;)
        {
            magics[square].magic = 0;
            // Require the top byte of (mask*magic) to be well-spread, else the magic is poor.
            while (popcount((mask * magics[square].magic) >> 56) < 6)
            {
                magics[square].magic = prng.sparse();
            }

            epochCounter++;
            for (subsetIndex = 0; subsetIndex < subsetCount; subsetIndex++)
            {
                unsigned tableIndex = magics[square].index(occupancy[subsetIndex]);
                if (epoch[tableIndex] < epochCounter)
                {
                    epoch[tableIndex]      = epochCounter;
                    attackBase[tableIndex] = reference[subsetIndex];
                }
                else if (attackBase[tableIndex] != reference[subsetIndex])
                {
                    break; // index collision with a different attack set -> reject this magic
                }
            }
        }
        attackBase += subsetCount;
    }
}

} // namespace

Bitboard bishop_attacks(int square, Bitboard occupancy)
{
    const Magic &entry = BishopMagics[square];
    return entry.attacks[entry.index(occupancy)];
}

Bitboard rook_attacks(int square, Bitboard occupancy)
{
    const Magic &entry = RookMagics[square];
    return entry.attacks[entry.index(occupancy)];
}

void init_bitboards()
{
    for (int square = 0; square < 64; square++)
    {
        Bitboard squareBit         = sq_bb(square);
        PawnAttacks[WHITE][square] = shift<NE>(squareBit) | shift<NW>(squareBit);
        PawnAttacks[BLACK][square] = shift<SE>(squareBit) | shift<SW>(squareBit);

        // Knight: all (±1,±2)/(±2,±1) offsets, rejecting wraps by file/rank distance.
        Bitboard  knightAttack = 0, kingAttack = 0;
        int       file = file_of(square), rank = rank_of(square);
        const int knightFile[8] = {1, 2, 2, 1, -1, -2, -2, -1};
        const int knightRank[8] = {2, 1, -1, -2, -2, -1, 1, 2};
        const int kingFile[8]   = {0, 1, 1, 1, 0, -1, -1, -1};
        const int kingRank[8]   = {1, 1, 0, -1, -1, -1, 0, 1};
        for (int offset = 0; offset < 8; offset++)
        {
            int targetFile = file + knightFile[offset], targetRank = rank + knightRank[offset];
            if (targetFile >= 0 && targetFile < 8 && targetRank >= 0 && targetRank < 8)
            {
                knightAttack |= sq_bb(make_square(targetFile, targetRank));
            }
            targetFile = file + kingFile[offset], targetRank = rank + kingRank[offset];
            if (targetFile >= 0 && targetFile < 8 && targetRank >= 0 && targetRank < 8)
            {
                kingAttack |= sq_bb(make_square(targetFile, targetRank));
            }
        }
        KnightAttacks[square] = knightAttack;
        KingAttacks[square]   = kingAttack;
    }

    init_magics(false, BishopTable, BishopMagics, BishopDirs);
    init_magics(true, RookTable, RookMagics, RookDirs);

    // BetweenBB: squares strictly between a and b when they share a rank/file/diagonal.
    // LineBB: the whole rank/file/diagonal through a and b (endpoints included), 0 if not aligned.
    for (int from = 0; from < 64; from++)
    {
        for (int to = 0; to < 64; to++)
        {
            BetweenBB[from][to] = 0;
            LineBB[from][to]    = 0;
            if (from == to)
            {
                continue;
            }
            for (auto attacker : {&rook_attacks, &bishop_attacks})
            {
                if (attacker(from, 0) & sq_bb(to))
                {
                    BetweenBB[from][to] = attacker(from, sq_bb(to)) & attacker(to, sq_bb(from));
                    LineBB[from][to]    = (sq_bb(from) | attacker(from, 0)) & (sq_bb(to) | attacker(to, 0));
                }
            }
        }
    }
}
