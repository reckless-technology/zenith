#pragma once
// Core value types: colours, pieces, squares, and the packed 16-bit move.
#include <bit>
#include <cstdint>
#include <string>

using Bitboard = uint64_t;

enum Color : int
{
    WHITE,
    BLACK,
    COLOR_NB = 2
};

inline Color operator~(Color c)
{
    return Color(c ^ 1);
}

enum PieceType : int
{
    PAWN,
    KNIGHT,
    BISHOP,
    ROOK,
    QUEEN,
    KING,
    PIECE_TYPE_NB = 6,
    NO_PIECE_TYPE = 6
};

// Mailbox piece code = colour*6 + type; NO_PIECE = 12.
enum Piece : int
{
    NO_PIECE = 12
};

inline Piece make_piece(Color c, PieceType pt)
{
    return Piece(c * 6 + pt);
}

inline Color color_of(Piece p)
{
    return Color(p / 6);
}

inline PieceType type_of(Piece p)
{
    return PieceType(p % 6);
}

// Squares: A1 = 0 … H8 = 63; rank = sq/8, file = sq%8.
enum : int
{
    NO_SQ     = 64,
    SQUARE_NB = 64
};

inline int rank_of(int sq)
{
    return sq >> 3;
}

inline int file_of(int sq)
{
    return sq & 7;
}

inline int make_square(int file, int rank)
{
    return rank * 8 + file;
}

inline int relative_rank(Color c, int sq)
{
    return c == WHITE ? rank_of(sq) : 7 - rank_of(sq);
}

inline std::string sq_name(int sq)
{
    if (sq >= 64)
    {
        return "-";
    }
    return std::string{char('a' + file_of(sq)), char('1' + rank_of(sq))};
}

// Castling rights bitmask.
enum : uint8_t
{
    CR_WK  = 1,
    CR_WQ  = 2,
    CR_BK  = 4,
    CR_BQ  = 8,
    CR_ALL = 15
};

// Move flags (CPW encoding): from(0..5) | to(6..11) | flag(12..15).
enum : uint16_t
{
    FLAG_QUIET       = 0,
    FLAG_DOUBLE      = 1,
    FLAG_KCASTLE     = 2,
    FLAG_QCASTLE     = 3,
    FLAG_CAPTURE     = 4,
    FLAG_EP          = 5,
    FLAG_PROMO_N     = 8,
    FLAG_PROMO_B     = 9,
    FLAG_PROMO_R     = 10,
    FLAG_PROMO_Q     = 11,
    FLAG_PROMO_CAP_N = 12,
    FLAG_PROMO_CAP_Q = 15,
};

class Move
{
    uint16_t d;

  public:
    Move() : d(0)
    {
    }

    explicit Move(uint16_t x) : d(x)
    {
    }

    Move(int from, int to, uint16_t flag) : d(uint16_t(from | (to << 6) | (flag << 12)))
    {
    }

    static Move none()
    {
        return Move(uint16_t(0));
    }

    uint16_t raw() const
    {
        return d;
    }

    int from() const
    {
        return d & 0x3f;
    }

    int to() const
    {
        return (d >> 6) & 0x3f;
    }

    uint16_t flag() const
    {
        return d >> 12;
    }

    bool is_none() const
    {
        return d == 0;
    }

    bool operator==(Move m) const
    {
        return d == m.d;
    }

    bool operator!=(Move m) const
    {
        return d != m.d;
    }

    bool is_capture() const
    {
        return flag() & 4;
    }

    bool is_ep() const
    {
        return flag() == FLAG_EP;
    }

    bool is_castle() const
    {
        return flag() == FLAG_KCASTLE || flag() == FLAG_QCASTLE;
    }

    bool is_double() const
    {
        return flag() == FLAG_DOUBLE;
    }

    bool is_promo() const
    {
        return flag() >= FLAG_PROMO_N;
    }

    bool is_quiet() const
    {
        return !is_capture() && !is_promo();
    }

    PieceType promo_pt() const
    {
        return PieceType(KNIGHT + (flag() & 3));
    }

    std::string to_uci() const
    {
        if (is_none())
        {
            return "0000";
        }
        std::string s = sq_name(from()) + sq_name(to());
        if (is_promo())
        {
            s += "  nbrq"[promo_pt() + 1]; // KNIGHT..QUEEN -> n,b,r,q
        }
        return s;
    }
};

// Search value scale.
constexpr int MAX_PLY           = 128;
constexpr int VALUE_INF         = 32001;
constexpr int VALUE_MATE        = 32000;
constexpr int VALUE_NONE        = 32002;
constexpr int VALUE_MATE_IN_MAX = VALUE_MATE - MAX_PLY; // scores at/above this are forced mates

inline bool is_mate_score(int v)
{
    return v >= VALUE_MATE_IN_MAX || v <= -VALUE_MATE_IN_MAX;
}

// Bitboard helpers (branch-free via <bit>).
constexpr Bitboard FILE_A = 0x0101010101010101ULL;
constexpr Bitboard FILE_B = FILE_A << 1;
constexpr Bitboard FILE_G = FILE_A << 6;
constexpr Bitboard FILE_H = FILE_A << 7;
constexpr Bitboard RANK_1 = 0xffULL;
constexpr Bitboard RANK_2 = RANK_1 << 8;
constexpr Bitboard RANK_4 = RANK_1 << 24;
constexpr Bitboard RANK_5 = RANK_1 << 32;
constexpr Bitboard RANK_7 = RANK_1 << 48;
constexpr Bitboard RANK_8 = RANK_1 << 56;

inline Bitboard sq_bb(int sq)
{
    return 1ULL << sq;
}

inline int popcount(Bitboard b)
{
    return std::popcount(b);
}

inline int lsb(Bitboard b)
{
    return std::countr_zero(b);
}

inline int pop_lsb(Bitboard &b)
{
    int s = lsb(b);
    b &= b - 1;
    return s;
}

inline bool more_than_one(Bitboard b)
{
    return b & (b - 1);
}

// Directional shifts (compass; wrap-safe via file masks).
enum Dir : int
{
    NORTH = 8,
    SOUTH = -8,
    EAST  = 1,
    WEST  = -1,
    NE    = 9,
    NW    = 7,
    SE    = -7,
    SW    = -9
};

template <int D> inline Bitboard shift(Bitboard b)
{
    if constexpr (D == NORTH)
    {
        return b << 8;
    }
    else if constexpr (D == SOUTH)
    {
        return b >> 8;
    }
    else if constexpr (D == EAST)
    {
        return (b & ~FILE_H) << 1;
    }
    else if constexpr (D == WEST)
    {
        return (b & ~FILE_A) >> 1;
    }
    else if constexpr (D == NE)
    {
        return (b & ~FILE_H) << 9;
    }
    else if constexpr (D == NW)
    {
        return (b & ~FILE_A) << 7;
    }
    else if constexpr (D == SE)
    {
        return (b & ~FILE_H) >> 7;
    }
    else if constexpr (D == SW)
    {
        return (b & ~FILE_A) >> 9;
    }
}
