#include "position.h"
#include "bitboard.h"
#include "nnue.h"
#include <cctype>
#include <sstream>

namespace Zobrist
{
uint64_t Piece[12][64];
uint64_t Castle[16];
uint64_t EpFile[8];
uint64_t Side;
} // namespace Zobrist

void init_zobrist()
{
    uint64_t state      = 0x9E3779B97F4A7C15ULL;
    auto     nextRandom = [&]() {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        return state * 0x2545F4914F6CDD1DULL;
    };
    for (int piece = 0; piece < 12; piece++)
    {
        for (int square = 0; square < 64; square++)
        {
            Zobrist::Piece[piece][square] = nextRandom();
        }
    }
    for (int castleRights = 0; castleRights < 16; castleRights++)
    {
        Zobrist::Castle[castleRights] = nextRandom();
    }
    for (int file = 0; file < 8; file++)
    {
        Zobrist::EpFile[file] = nextRandom();
    }
    Zobrist::Side = nextRandom();
}

// castleMask[sq]: rights to KEEP when a piece leaves or arrives on sq (AND-ed into castling).
static uint8_t CastleMask[64];
static bool    castleMaskInit = [] {
    for (int i = 0; i < 64; i++)
    {
        CastleMask[i] = CR_ALL;
    }
    CastleMask[make_square(4, 0)] = uint8_t(~(CR_WK | CR_WQ)); // e1
    CastleMask[make_square(0, 0)] = uint8_t(~CR_WQ);           // a1
    CastleMask[make_square(7, 0)] = uint8_t(~CR_WK);           // h1
    CastleMask[make_square(4, 7)] = uint8_t(~(CR_BK | CR_BQ)); // e8
    CastleMask[make_square(0, 7)] = uint8_t(~CR_BQ);           // a8
    CastleMask[make_square(7, 7)] = uint8_t(~CR_BK);           // h8
    return true;
}();

void Position::put(Color color, PieceType pieceType, int square)
{
    Bitboard squareBit = sq_bb(square);
    byColor[color] |= squareBit;
    byType[pieceType] |= squareBit;
    Piece piece   = make_piece(color, pieceType);
    board[square] = piece;
    key ^= Zobrist::Piece[piece][square];
    if (pieceType == PAWN)
    {
        pawnKey ^= Zobrist::Piece[piece][square];
    }
    if (nnue::is_loaded())
    {
        nnue::add_feature(acc, color, pieceType, square);
    }
}

void Position::remove(int square)
{
    Piece    piece     = board[square];
    Bitboard squareBit = sq_bb(square);
    byColor[color_of(piece)] ^= squareBit;
    byType[type_of(piece)] ^= squareBit;
    key ^= Zobrist::Piece[piece][square];
    if (type_of(piece) == PAWN)
    {
        pawnKey ^= Zobrist::Piece[piece][square];
    }
    board[square] = NO_PIECE;
    if (nnue::is_loaded())
    {
        nnue::remove_feature(acc, color_of(piece), type_of(piece), square);
    }
}

void Position::move_piece(int from, int to)
{
    Piece    piece   = board[from];
    Bitboard fromBit = sq_bb(from), toBit = sq_bb(to);
    Bitboard fromToBits = fromBit | toBit;
    byColor[color_of(piece)] ^= fromToBits;
    byType[type_of(piece)] ^= fromToBits;
    key ^= Zobrist::Piece[piece][from] ^ Zobrist::Piece[piece][to];
    if (type_of(piece) == PAWN)
    {
        pawnKey ^= Zobrist::Piece[piece][from] ^ Zobrist::Piece[piece][to];
    }
    board[to]   = piece;
    board[from] = NO_PIECE;
    if (nnue::is_loaded())
    {
        nnue::move_feature(acc, color_of(piece), type_of(piece), from, to);
    }
}

Bitboard Position::attackers_to(int square, Color color, Bitboard occupancy) const
{
    Bitboard attackers = 0;
    attackers |= pawn_attacks(~color, square) & pieces(color, PAWN);
    attackers |= knight_attacks(square) & pieces(color, KNIGHT);
    attackers |= king_attacks(square) & pieces(color, KING);
    attackers |= bishop_attacks(square, occupancy) & (pieces(color, BISHOP) | pieces(color, QUEEN));
    attackers |= rook_attacks(square, occupancy) & (pieces(color, ROOK) | pieces(color, QUEEN));
    return attackers;
}

void Position::make_move(Move move)
{
    Color     side = stm, opponent = ~stm;
    int       from = move.from(), to = move.to();
    PieceType pieceType = type_of(board[from]);

    if (epSq != NO_SQ)
    {
        key ^= Zobrist::EpFile[file_of(epSq)];
        epSq = NO_SQ;
    }
    halfmove++;

    if (move.is_capture())
    {
        halfmove = 0;
        if (move.is_ep())
        {
            remove(to + (side == WHITE ? -8 : 8)); // captured pawn sits behind the ep target
        }
        else
        {
            remove(to);
        }
    }

    if (move.is_castle())
    {
        int rank = rank_of(from);
        if (move.flag() == FLAG_KCASTLE)
        {
            move_piece(make_square(7, rank), make_square(5, rank));
        }
        else
        {
            move_piece(make_square(0, rank), make_square(3, rank));
        }
    }

    move_piece(from, to);

    if (pieceType == PAWN)
    {
        halfmove = 0;
        if (move.is_promo())
        {
            remove(to);
            put(side, move.promo_pt(), to);
        }
        else if (move.is_double())
        {
            int epSquare = (from + to) / 2;
            if (pawn_attacks(side, epSquare) & pieces(opponent, PAWN))
            {
                epSq = epSquare;
                key ^= Zobrist::EpFile[file_of(epSquare)];
            }
        }
    }

    // King-input buckets: a king move (including castling) can change the moving side's king bucket, which
    // shifts that whole perspective's feature block — refresh it. The primitives above updated both
    // perspectives incrementally with the pre-move buckets; refresh_perspective discards the stale own half.
    if (pieceType == KING && nnue::is_loaded())
    {
        nnue::update_king_bucket(acc, *this, side);
    }

    uint8_t oldCastling = castling;
    castling &= CastleMask[from] & CastleMask[to];
    if (castling != oldCastling)
    {
        key ^= Zobrist::Castle[oldCastling] ^ Zobrist::Castle[castling];
    }

    if (side == BLACK)
    {
        fullmove++;
    }
    stm = opponent;
    key ^= Zobrist::Side;
    ply++;
}

void Position::make_null()
{
    if (epSq != NO_SQ)
    {
        key ^= Zobrist::EpFile[file_of(epSq)];
        epSq = NO_SQ;
    }
    stm = ~stm;
    key ^= Zobrist::Side;
    halfmove++;
    ply++;
}

bool Position::is_legal(Move move) const
{
    Color    side = stm;
    Position copy = *this;
    copy.make_move(move);
    return !copy.attacked_by(copy.king_sq(side), ~side);
}

Bitboard Position::pinned_to_king() const
{
    Color    side = stm, opponent = ~side;
    int      kingSquare   = king_sq(side);
    Bitboard occupancy    = occupied();
    Bitboard pinnedPieces = 0;
    // Enemy sliders that would hit our king on an empty board are candidate pinners.
    Bitboard snipers = (rook_attacks(kingSquare, 0) & (pieces(opponent, ROOK) | pieces(opponent, QUEEN))) |
                       (bishop_attacks(kingSquare, 0) & (pieces(opponent, BISHOP) | pieces(opponent, QUEEN)));
    while (snipers)
    {
        int      sniperSquare = pop_lsb(snipers);
        Bitboard between      = between_bb(kingSquare, sniperSquare) & occupancy;
        // Exactly one piece between the sniper and our king, and it is ours => that piece is pinned.
        if (between && !(between & (between - 1)) && (between & byColor[side]))
        {
            pinnedPieces |= between;
        }
    }
    return pinnedPieces;
}

bool Position::is_legal_fast(Move move, Bitboard checkers, Bitboard pinned) const
{
    Color side = stm, opponent = ~side;
    int   from = move.from(), to = move.to();
    int   kingSquare = king_sq(side);

    // Castling is generated fully legal by movegen; en passant can expose the king along a rank (rare).
    // Defer both to the exact copy-make test.
    if (move.is_castle() || move.is_ep())
    {
        return is_legal(move);
    }

    // King move: the destination must be unattacked once the king vacates (so a slider sees through it).
    if (from == kingSquare)
    {
        return !attackers_to(to, opponent, occupied() ^ sq_bb(kingSquare));
    }

    // In check: double check leaves only king moves; single check requires capturing the checker or blocking
    // the ray between it and the king.
    if (checkers)
    {
        if (checkers & (checkers - 1))
        {
            return false; // double check, and this is not a king move
        }
        int checkerSquare = lsb(checkers);
        if (!(sq_bb(to) & (checkers | between_bb(kingSquare, checkerSquare))))
        {
            return false;
        }
    }

    // A pinned piece may only move along the pin ray (the line through our king and the piece).
    if ((pinned & sq_bb(from)) && !(line_bb(kingSquare, from) & sq_bb(to)))
    {
        return false;
    }
    return true;
}

bool Position::gives_check(Move move) const
{
    Position copy = *this;
    copy.make_move(move);
    return copy.in_check();
}

void Position::set_fen(const std::string &fen)
{
    for (int color = 0; color < COLOR_NB; color++)
    {
        byColor[color] = 0;
    }
    for (int pieceType = 0; pieceType < PIECE_TYPE_NB; pieceType++)
    {
        byType[pieceType] = 0;
    }
    for (int square = 0; square < 64; square++)
    {
        board[square] = NO_PIECE;
    }
    key      = 0;
    pawnKey  = 0;
    castling = 0;
    epSq     = NO_SQ;
    halfmove = 0;
    fullmove = 1;
    ply      = 0;
    stm      = WHITE;

    std::istringstream stream(fen);
    std::string        boardStr, side, castleStr, epStr;
    stream >> boardStr >> side >> castleStr >> epStr;
    int halfmoveClock = 0, fullmoveNumber = 1;
    stream >> halfmoveClock >> fullmoveNumber;

    int file = 0, rank = 7;
    for (char ch : boardStr)
    {
        if (ch == '/')
        {
            rank--;
            file = 0;
        }
        else if (isdigit((unsigned char)ch))
        {
            file += ch - '0';
        }
        else
        {
            Color     color = isupper((unsigned char)ch) ? WHITE : BLACK;
            PieceType pieceType;
            switch (tolower((unsigned char)ch))
            {
            case 'p':
                pieceType = PAWN;
                break;
            case 'n':
                pieceType = KNIGHT;
                break;
            case 'b':
                pieceType = BISHOP;
                break;
            case 'r':
                pieceType = ROOK;
                break;
            case 'q':
                pieceType = QUEEN;
                break;
            default:
                pieceType = KING;
                break;
            }
            put(color, pieceType, make_square(file, rank));
            file++;
        }
    }

    stm = (side == "b") ? BLACK : WHITE;
    for (char ch : castleStr)
    {
        switch (ch)
        {
        case 'K':
            castling |= CR_WK;
            break;
        case 'Q':
            castling |= CR_WQ;
            break;
        case 'k':
            castling |= CR_BK;
            break;
        case 'q':
            castling |= CR_BQ;
            break;
        default:
            break;
        }
    }
    if (epStr != "-" && epStr.size() >= 2)
    {
        int epSquare = make_square(epStr[0] - 'a', epStr[1] - '1');
        // Keep the ep square only when a pawn of the side to move can actually capture there — matches
        // make_move, so equal positions hash equally regardless of how they were reached.
        if (pawn_attacks(~stm, epSquare) & pieces(stm, PAWN))
        {
            epSq = epSquare;
        }
    }
    halfmove = halfmoveClock;
    fullmove = fullmoveNumber;

    if (stm == BLACK)
    {
        key ^= Zobrist::Side;
    }
    key ^= Zobrist::Castle[castling];
    if (epSq != NO_SQ)
    {
        key ^= Zobrist::EpFile[file_of(epSq)];
    }
    // Authoritative accumulator rebuild (put() updated it incrementally from an uninitialised state above).
    if (nnue::is_loaded())
    {
        nnue::refresh(acc, *this);
    }
}

std::string Position::fen() const
{
    std::string result;
    for (int rank = 7; rank >= 0; rank--)
    {
        int emptyCount = 0;
        for (int file = 0; file < 8; file++)
        {
            Piece piece = board[make_square(file, rank)];
            if (piece == NO_PIECE)
            {
                emptyCount++;
                continue;
            }
            if (emptyCount)
            {
                result += char('0' + emptyCount);
                emptyCount = 0;
            }
            const char *names = "PNBRQKpnbrqk";
            result += names[piece];
        }
        if (emptyCount)
        {
            result += char('0' + emptyCount);
        }
        if (rank)
        {
            result += '/';
        }
    }
    result += stm == WHITE ? " w " : " b ";
    std::string castleStr;
    if (castling & CR_WK)
    {
        castleStr += 'K';
    }
    if (castling & CR_WQ)
    {
        castleStr += 'Q';
    }
    if (castling & CR_BK)
    {
        castleStr += 'k';
    }
    if (castling & CR_BQ)
    {
        castleStr += 'q';
    }
    result += castleStr.empty() ? "-" : castleStr;
    result += ' ';
    result += epSq == NO_SQ ? "-" : sq_name(epSq);
    result += ' ' + std::to_string(halfmove) + ' ' + std::to_string(fullmove);
    return result;
}
