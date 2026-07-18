#include "bitboard.h"

Bitboard PawnAttacks[COLOR_NB][64];
Bitboard KnightAttacks[64];
Bitboard KingAttacks[64];
Bitboard BetweenBB[64][64];

namespace
{

// Deterministic sparse PRNG for magic search (xorshift64*), seeded once.
struct PRNG
{
    uint64_t s;

    explicit PRNG(uint64_t seed) : s(seed)
    {
    }

    uint64_t next()
    {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        return s * 0x2545F4914F6CDD1DULL;
    }

    uint64_t sparse()
    {
        return next() & next() & next();
    } // few set bits -> good magic candidates
};

// Ray-walk sliding attacks (used to build masks and to fill the magic tables).
Bitboard sliding_attack(int sq, Bitboard occ, const int deltas[4])
{
    Bitboard att = 0;
    for (int i = 0; i < 4; i++)
    {
        int d = deltas[i];
        int s = sq;
        while (true)
        {
            int prevFile = file_of(s);
            int ns       = s + d;
            if (ns < 0 || ns >= 64)
            {
                break;
            }
            // reject rank/file wraps: any king-step move changes file by at most 1
            if (std::abs(file_of(ns) - prevFile) > 1)
            {
                break;
            }
            att |= sq_bb(ns);
            if (occ & sq_bb(ns))
            {
                break; // blocker occupies this square; stop after including it
            }
            s = ns;
        }
    }
    return att;
}

const int RookDirs[4]   = {NORTH, SOUTH, EAST, WEST};
const int BishopDirs[4] = {NE, NW, SE, SW};

struct Magic
{
    Bitboard  mask    = 0;
    Bitboard  magic   = 0;
    Bitboard *attacks = nullptr;
    unsigned  shift   = 0;

    unsigned index(Bitboard occ) const
    {
        return unsigned(((occ & mask) * magic) >> shift);
    }
};

Magic    RookMagics[64];
Magic    BishopMagics[64];
Bitboard RookTable[0x19000];  // 102400
Bitboard BishopTable[0x1480]; // 5248

void init_magics(bool rook, Bitboard *table, Magic magics[64], const int deltas[4])
{
    Bitboard  occupancy[4096], reference[4096];
    int       epoch[4096] = {0};
    int       cnt         = 0;
    Bitboard *base        = table;

    for (int sq = 0; sq < 64; sq++)
    {
        // Relevant-occupancy mask = empty-board rays minus the edges not on the piece's own rank/file.
        Bitboard edges     = ((RANK_1 | RANK_8) & ~rank_bb(sq)) | ((FILE_A | FILE_H) & ~file_bb(sq));
        Bitboard mask      = sliding_attack(sq, 0, deltas) & ~edges;
        unsigned bits      = popcount(mask);
        magics[sq].mask    = mask;
        magics[sq].shift   = 64 - bits;
        magics[sq].attacks = base;

        // Enumerate every subset of mask (Carry-Rippler), recording its true attack set.
        Bitboard b    = 0;
        int      size = 0;
        do
        {
            occupancy[size] = b;
            reference[size] = sliding_attack(sq, b, deltas);
            size++;
            b = (b - mask) & mask;
        } while (b);

        PRNG prng(0x9E3779B97F4A7C15ULL ^ (uint64_t(sq) * 0xBF58476D1CE4E5B9ULL) ^ (rook ? 1 : 2));
        for (int i = 0; i < size;)
        {
            magics[sq].magic = 0;
            // Require the top byte of (mask*magic) to be well-spread, else the magic is poor.
            while (popcount((mask * magics[sq].magic) >> 56) < 6)
            {
                magics[sq].magic = prng.sparse();
            }

            cnt++;
            for (i = 0; i < size; i++)
            {
                unsigned idx = magics[sq].index(occupancy[i]);
                if (epoch[idx] < cnt)
                {
                    epoch[idx] = cnt;
                    base[idx]  = reference[i];
                }
                else if (base[idx] != reference[i])
                {
                    break; // index collision with a different attack set -> reject this magic
                }
            }
        }
        base += size;
    }
}

} // namespace

Bitboard bishop_attacks(int sq, Bitboard occ)
{
    const Magic &m = BishopMagics[sq];
    return m.attacks[m.index(occ)];
}

Bitboard rook_attacks(int sq, Bitboard occ)
{
    const Magic &m = RookMagics[sq];
    return m.attacks[m.index(occ)];
}

void init_bitboards()
{
    for (int sq = 0; sq < 64; sq++)
    {
        Bitboard b             = sq_bb(sq);
        PawnAttacks[WHITE][sq] = shift<NE>(b) | shift<NW>(b);
        PawnAttacks[BLACK][sq] = shift<SE>(b) | shift<SW>(b);

        // Knight: all (±1,±2)/(±2,±1) offsets, rejecting wraps by file/rank distance.
        Bitboard  n = 0, k = 0;
        int       f = file_of(sq), r = rank_of(sq);
        const int nf[8] = {1, 2, 2, 1, -1, -2, -2, -1};
        const int nr[8] = {2, 1, -1, -2, -2, -1, 1, 2};
        const int kf[8] = {0, 1, 1, 1, 0, -1, -1, -1};
        const int kr[8] = {1, 1, 0, -1, -1, -1, 0, 1};
        for (int i = 0; i < 8; i++)
        {
            int tf = f + nf[i], tr = r + nr[i];
            if (tf >= 0 && tf < 8 && tr >= 0 && tr < 8)
            {
                n |= sq_bb(make_square(tf, tr));
            }
            tf = f + kf[i], tr = r + kr[i];
            if (tf >= 0 && tf < 8 && tr >= 0 && tr < 8)
            {
                k |= sq_bb(make_square(tf, tr));
            }
        }
        KnightAttacks[sq] = n;
        KingAttacks[sq]   = k;
    }

    init_magics(false, BishopTable, BishopMagics, BishopDirs);
    init_magics(true, RookTable, RookMagics, RookDirs);

    // BetweenBB: squares strictly between a and b when they share a rank/file/diagonal.
    for (int a = 0; a < 64; a++)
    {
        for (int b = 0; b < 64; b++)
        {
            BetweenBB[a][b] = 0;
            if (a == b)
            {
                continue;
            }
            for (auto attacker : {&rook_attacks, &bishop_attacks})
            {
                if (attacker(a, 0) & sq_bb(b))
                {
                    BetweenBB[a][b] = attacker(a, sq_bb(b)) & attacker(b, sq_bb(a));
                }
            }
        }
    }
}
