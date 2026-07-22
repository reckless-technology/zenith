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
    uint64_t s   = 0x9E3779B97F4A7C15ULL;
    auto     rnd = [&]() {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        return s * 0x2545F4914F6CDD1DULL;
    };
    for (int p = 0; p < 12; p++)
    {
        for (int sq = 0; sq < 64; sq++)
        {
            Zobrist::Piece[p][sq] = rnd();
        }
    }
    for (int i = 0; i < 16; i++)
    {
        Zobrist::Castle[i] = rnd();
    }
    for (int f = 0; f < 8; f++)
    {
        Zobrist::EpFile[f] = rnd();
    }
    Zobrist::Side = rnd();
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

void Position::put(Color c, PieceType pt, int sq)
{
    Bitboard b = sq_bb(sq);
    byColor[c] |= b;
    byType[pt] |= b;
    Piece p   = make_piece(c, pt);
    board[sq] = p;
    key ^= Zobrist::Piece[p][sq];
    if (pt == PAWN)
    {
        pawnKey ^= Zobrist::Piece[p][sq];
    }
    if (nnue::is_loaded())
    {
        nnue::add_feature(acc, c, pt, sq);
    }
}

void Position::remove(int sq)
{
    Piece    p = board[sq];
    Bitboard b = sq_bb(sq);
    byColor[color_of(p)] ^= b;
    byType[type_of(p)] ^= b;
    key ^= Zobrist::Piece[p][sq];
    if (type_of(p) == PAWN)
    {
        pawnKey ^= Zobrist::Piece[p][sq];
    }
    board[sq] = NO_PIECE;
    if (nnue::is_loaded())
    {
        nnue::remove_feature(acc, color_of(p), type_of(p), sq);
    }
}

void Position::move_piece(int from, int to)
{
    Piece    p  = board[from];
    Bitboard fb = sq_bb(from), tb = sq_bb(to);
    Bitboard ft = fb | tb;
    byColor[color_of(p)] ^= ft;
    byType[type_of(p)] ^= ft;
    key ^= Zobrist::Piece[p][from] ^ Zobrist::Piece[p][to];
    if (type_of(p) == PAWN)
    {
        pawnKey ^= Zobrist::Piece[p][from] ^ Zobrist::Piece[p][to];
    }
    board[to]   = p;
    board[from] = NO_PIECE;
    if (nnue::is_loaded())
    {
        nnue::move_feature(acc, color_of(p), type_of(p), from, to);
    }
}

Bitboard Position::attackers_to(int sq, Color c, Bitboard occ) const
{
    Bitboard att = 0;
    att |= pawn_attacks(~c, sq) & pieces(c, PAWN);
    att |= knight_attacks(sq) & pieces(c, KNIGHT);
    att |= king_attacks(sq) & pieces(c, KING);
    att |= bishop_attacks(sq, occ) & (pieces(c, BISHOP) | pieces(c, QUEEN));
    att |= rook_attacks(sq, occ) & (pieces(c, ROOK) | pieces(c, QUEEN));
    return att;
}

void Position::make_move(Move m)
{
    Color     us = stm, them = ~stm;
    int       from = m.from(), to = m.to();
    PieceType pt = type_of(board[from]);

    if (epSq != NO_SQ)
    {
        key ^= Zobrist::EpFile[file_of(epSq)];
        epSq = NO_SQ;
    }
    halfmove++;

    if (m.is_capture())
    {
        halfmove = 0;
        if (m.is_ep())
        {
            remove(to + (us == WHITE ? -8 : 8)); // captured pawn sits behind the ep target
        }
        else
        {
            remove(to);
        }
    }

    if (m.is_castle())
    {
        int r = rank_of(from);
        if (m.flag() == FLAG_KCASTLE)
        {
            move_piece(make_square(7, r), make_square(5, r));
        }
        else
        {
            move_piece(make_square(0, r), make_square(3, r));
        }
    }

    move_piece(from, to);

    if (pt == PAWN)
    {
        halfmove = 0;
        if (m.is_promo())
        {
            remove(to);
            put(us, m.promo_pt(), to);
        }
        else if (m.is_double())
        {
            int ep = (from + to) / 2;
            if (pawn_attacks(us, ep) & pieces(them, PAWN))
            {
                epSq = ep;
                key ^= Zobrist::EpFile[file_of(ep)];
            }
        }
    }

    // King-input buckets: a king move (including castling) can change the moving side's king bucket, which
    // shifts that whole perspective's feature block — refresh it. The primitives above updated both
    // perspectives incrementally with the pre-move buckets; refresh_perspective discards the stale own half.
    if (pt == KING && nnue::is_loaded())
    {
        nnue::update_king_bucket(acc, *this, us);
    }

    uint8_t oldCr = castling;
    castling &= CastleMask[from] & CastleMask[to];
    if (castling != oldCr)
    {
        key ^= Zobrist::Castle[oldCr] ^ Zobrist::Castle[castling];
    }

    if (us == BLACK)
    {
        fullmove++;
    }
    stm = them;
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

bool Position::is_legal(Move m) const
{
    Color    us = stm;
    Position c  = *this;
    c.make_move(m);
    return !c.attacked_by(c.king_sq(us), ~us);
}

Bitboard Position::pinned_to_king() const
{
    Color    us = stm, them = ~us;
    int      ksq = king_sq(us);
    Bitboard occ = occupied();
    Bitboard out = 0;
    // Enemy sliders that would hit our king on an empty board are candidate pinners.
    Bitboard snipers = (rook_attacks(ksq, 0) & (pieces(them, ROOK) | pieces(them, QUEEN))) |
                       (bishop_attacks(ksq, 0) & (pieces(them, BISHOP) | pieces(them, QUEEN)));
    while (snipers)
    {
        int      s       = pop_lsb(snipers);
        Bitboard between = between_bb(ksq, s) & occ;
        // Exactly one piece between the sniper and our king, and it is ours => that piece is pinned.
        if (between && !(between & (between - 1)) && (between & byColor[us]))
        {
            out |= between;
        }
    }
    return out;
}

bool Position::is_legal_fast(Move m, Bitboard checkers, Bitboard pinned) const
{
    Color us = stm, them = ~us;
    int   from = m.from(), to = m.to();
    int   ksq = king_sq(us);

    // Castling is generated fully legal by movegen; en passant can expose the king along a rank (rare).
    // Defer both to the exact copy-make test.
    if (m.is_castle() || m.is_ep())
    {
        return is_legal(m);
    }

    // King move: the destination must be unattacked once the king vacates (so a slider sees through it).
    if (from == ksq)
    {
        return !attackers_to(to, them, occupied() ^ sq_bb(ksq));
    }

    // In check: double check leaves only king moves; single check requires capturing the checker or blocking
    // the ray between it and the king.
    if (checkers)
    {
        if (checkers & (checkers - 1))
        {
            return false; // double check, and this is not a king move
        }
        int checkerSq = lsb(checkers);
        if (!(sq_bb(to) & (checkers | between_bb(ksq, checkerSq))))
        {
            return false;
        }
    }

    // A pinned piece may only move along the pin ray (the line through our king and the piece).
    if ((pinned & sq_bb(from)) && !(line_bb(ksq, from) & sq_bb(to)))
    {
        return false;
    }
    return true;
}

bool Position::gives_check(Move m) const
{
    Position c = *this;
    c.make_move(m);
    return c.in_check();
}

void Position::set_fen(const std::string &fen)
{
    for (int i = 0; i < COLOR_NB; i++)
    {
        byColor[i] = 0;
    }
    for (int i = 0; i < PIECE_TYPE_NB; i++)
    {
        byType[i] = 0;
    }
    for (int i = 0; i < 64; i++)
    {
        board[i] = NO_PIECE;
    }
    key      = 0;
    pawnKey  = 0;
    castling = 0;
    epSq     = NO_SQ;
    halfmove = 0;
    fullmove = 1;
    ply      = 0;
    stm      = WHITE;

    std::istringstream ss(fen);
    std::string        boardStr, side, castleStr, epStr;
    ss >> boardStr >> side >> castleStr >> epStr;
    int hm = 0, fm = 1;
    ss >> hm >> fm;

    int f = 0, r = 7;
    for (char ch : boardStr)
    {
        if (ch == '/')
        {
            r--;
            f = 0;
        }
        else if (isdigit((unsigned char)ch))
        {
            f += ch - '0';
        }
        else
        {
            Color     c = isupper((unsigned char)ch) ? WHITE : BLACK;
            PieceType pt;
            switch (tolower((unsigned char)ch))
            {
            case 'p':
                pt = PAWN;
                break;
            case 'n':
                pt = KNIGHT;
                break;
            case 'b':
                pt = BISHOP;
                break;
            case 'r':
                pt = ROOK;
                break;
            case 'q':
                pt = QUEEN;
                break;
            default:
                pt = KING;
                break;
            }
            put(c, pt, make_square(f, r));
            f++;
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
        int ep = make_square(epStr[0] - 'a', epStr[1] - '1');
        // Keep the ep square only when a pawn of the side to move can actually capture there — matches
        // make_move, so equal positions hash equally regardless of how they were reached.
        if (pawn_attacks(~stm, ep) & pieces(stm, PAWN))
        {
            epSq = ep;
        }
    }
    halfmove = hm;
    fullmove = fm;

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
    std::string s;
    for (int r = 7; r >= 0; r--)
    {
        int empty = 0;
        for (int f = 0; f < 8; f++)
        {
            Piece p = board[make_square(f, r)];
            if (p == NO_PIECE)
            {
                empty++;
                continue;
            }
            if (empty)
            {
                s += char('0' + empty);
                empty = 0;
            }
            const char *names = "PNBRQKpnbrqk";
            s += names[p];
        }
        if (empty)
        {
            s += char('0' + empty);
        }
        if (r)
        {
            s += '/';
        }
    }
    s += stm == WHITE ? " w " : " b ";
    std::string cr;
    if (castling & CR_WK)
    {
        cr += 'K';
    }
    if (castling & CR_WQ)
    {
        cr += 'Q';
    }
    if (castling & CR_BK)
    {
        cr += 'k';
    }
    if (castling & CR_BQ)
    {
        cr += 'q';
    }
    s += cr.empty() ? "-" : cr;
    s += ' ';
    s += epSq == NO_SQ ? "-" : sq_name(epSq);
    s += ' ' + std::to_string(halfmove) + ' ' + std::to_string(fullmove);
    return s;
}
