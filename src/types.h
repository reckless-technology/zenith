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

inline Piece make_piece(Color color, PieceType pieceType)
{
    return Piece(color * 6 + pieceType);
}

inline Color color_of(Piece piece)
{
    return Color(piece / 6);
}

inline PieceType type_of(Piece piece)
{
    return PieceType(piece % 6);
}

// Squares: A1 = 0 … H8 = 63; rank = sq/8, file = sq%8.
enum : int
{
    NO_SQ     = 64,
    SQUARE_NB = 64
};

inline int rank_of(int square)
{
    return square >> 3;
}

inline int file_of(int square)
{
    return square & 7;
}

inline int make_square(int file, int rank)
{
    return rank * 8 + file;
}

inline int relative_rank(Color color, int square)
{
    return color == WHITE ? rank_of(square) : 7 - rank_of(square);
}

inline std::string sq_name(int square)
{
    if (square >= 64)
    {
        return "-";
    }
    return std::string{char('a' + file_of(square)), char('1' + rank_of(square))};
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
    uint16_t data;

  public:
    Move() : data(0)
    {
    }

    explicit Move(uint16_t raw) : data(raw)
    {
    }

    Move(int from, int to, uint16_t flag) : data(uint16_t(from | (to << 6) | (flag << 12)))
    {
    }

    static Move none()
    {
        return Move(uint16_t(0));
    }

    uint16_t raw() const
    {
        return data;
    }

    int from() const
    {
        return data & 0x3f;
    }

    int to() const
    {
        return (data >> 6) & 0x3f;
    }

    uint16_t flag() const
    {
        return data >> 12;
    }

    bool is_none() const
    {
        return data == 0;
    }

    bool operator==(Move other) const
    {
        return data == other.data;
    }

    bool operator!=(Move other) const
    {
        return data != other.data;
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
        std::string uci = sq_name(from()) + sq_name(to());
        if (is_promo())
        {
            uci += "  nbrq"[promo_pt() + 1]; // KNIGHT..QUEEN -> n,b,r,q
        }
        return uci;
    }
};

// Search value scale.
constexpr int MAX_PLY           = 128;
constexpr int VALUE_INF         = 32001;
constexpr int VALUE_MATE        = 32000;
constexpr int VALUE_NONE        = 32002;
constexpr int VALUE_MATE_IN_MAX = VALUE_MATE - MAX_PLY; // scores at/above this are forced mates

inline bool is_mate_score(int value)
{
    return value >= VALUE_MATE_IN_MAX || value <= -VALUE_MATE_IN_MAX;
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

inline Bitboard sq_bb(int square)
{
    return 1ULL << square;
}

inline int popcount(Bitboard bitboard)
{
    return std::popcount(bitboard);
}

inline int lsb(Bitboard bitboard)
{
    return std::countr_zero(bitboard);
}

inline int pop_lsb(Bitboard &bitboard)
{
    int square = lsb(bitboard);
    bitboard &= bitboard - 1;
    return square;
}

inline bool more_than_one(Bitboard bitboard)
{
    return bitboard & (bitboard - 1);
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

template <int Direction> inline Bitboard shift(Bitboard bitboard)
{
    if constexpr (Direction == NORTH)
    {
        return bitboard << 8;
    }
    else if constexpr (Direction == SOUTH)
    {
        return bitboard >> 8;
    }
    else if constexpr (Direction == EAST)
    {
        return (bitboard & ~FILE_H) << 1;
    }
    else if constexpr (Direction == WEST)
    {
        return (bitboard & ~FILE_A) >> 1;
    }
    else if constexpr (Direction == NE)
    {
        return (bitboard & ~FILE_H) << 9;
    }
    else if constexpr (Direction == NW)
    {
        return (bitboard & ~FILE_A) << 7;
    }
    else if constexpr (Direction == SE)
    {
        return (bitboard & ~FILE_H) >> 7;
    }
    else if constexpr (Direction == SW)
    {
        return (bitboard & ~FILE_A) >> 9;
    }
}
