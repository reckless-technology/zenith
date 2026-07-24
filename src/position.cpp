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
    uint64_t state       = 0x9E3779B97F4A7C15ULL;
    auto     next_random = [&]() {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        return state * 0x2545F4914F6CDD1DULL;
    };
    for (int piece = 0; piece < 12; piece++)
    {
        for (int square = 0; square < 64; square++)
        {
            Zobrist::Piece[piece][square] = next_random();
        }
    }
    for (int castle_rights = 0; castle_rights < 16; castle_rights++)
    {
        Zobrist::Castle[castle_rights] = next_random();
    }
    for (int file = 0; file < 8; file++)
    {
        Zobrist::EpFile[file] = next_random();
    }
    Zobrist::Side = next_random();
}

// castleMask[sq]: rights to KEEP when a piece leaves or arrives on sq (AND-ed into castling).
static uint8_t CastleMask[64];
static bool    castle_mask_init = [] {
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

void Position::put(Color color, PieceType piece_type, int square)
{
    Bitboard square_bit = sq_bb(square);
    by_color[color] |= square_bit;
    by_type[piece_type] |= square_bit;
    Piece piece   = make_piece(color, piece_type);
    board[square] = piece;
    key ^= Zobrist::Piece[piece][square];
    if (piece_type == PAWN)
    {
        pawn_key ^= Zobrist::Piece[piece][square];
    }
    if (nnue::is_loaded())
    {
        nnue::add_feature(acc, color, piece_type, square);
    }
}

void Position::remove(int square)
{
    Piece    piece      = board[square];
    Bitboard square_bit = sq_bb(square);
    by_color[color_of(piece)] ^= square_bit;
    by_type[type_of(piece)] ^= square_bit;
    key ^= Zobrist::Piece[piece][square];
    if (type_of(piece) == PAWN)
    {
        pawn_key ^= Zobrist::Piece[piece][square];
    }
    board[square] = NO_PIECE;
    if (nnue::is_loaded())
    {
        nnue::remove_feature(acc, color_of(piece), type_of(piece), square);
    }
}

void Position::move_piece(int from, int to)
{
    Piece    piece    = board[from];
    Bitboard from_bit = sq_bb(from), to_bit = sq_bb(to);
    Bitboard from_to_bits = from_bit | to_bit;
    by_color[color_of(piece)] ^= from_to_bits;
    by_type[type_of(piece)] ^= from_to_bits;
    key ^= Zobrist::Piece[piece][from] ^ Zobrist::Piece[piece][to];
    if (type_of(piece) == PAWN)
    {
        pawn_key ^= Zobrist::Piece[piece][from] ^ Zobrist::Piece[piece][to];
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
    PieceType piece_type = type_of(board[from]);

    if (ep_sq != NO_SQ)
    {
        key ^= Zobrist::EpFile[file_of(ep_sq)];
        ep_sq = NO_SQ;
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

    if (piece_type == PAWN)
    {
        halfmove = 0;
        if (move.is_promo())
        {
            remove(to);
            put(side, move.promo_pt(), to);
        }
        else if (move.is_double())
        {
            int ep_square = (from + to) / 2;
            if (pawn_attacks(side, ep_square) & pieces(opponent, PAWN))
            {
                ep_sq = ep_square;
                key ^= Zobrist::EpFile[file_of(ep_square)];
            }
        }
    }

    // King-input buckets: a king move (including castling) can change the moving side's king bucket, which
    // shifts that whole perspective's feature block — refresh it. The primitives above updated both
    // perspectives incrementally with the pre-move buckets; refresh_perspective discards the stale own half.
    if (piece_type == KING && nnue::is_loaded())
    {
        nnue::update_king_bucket(acc, *this, side);
    }

    uint8_t old_castling = castling;
    castling &= CastleMask[from] & CastleMask[to];
    if (castling != old_castling)
    {
        key ^= Zobrist::Castle[old_castling] ^ Zobrist::Castle[castling];
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
    if (ep_sq != NO_SQ)
    {
        key ^= Zobrist::EpFile[file_of(ep_sq)];
        ep_sq = NO_SQ;
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
    int      king_square   = king_sq(side);
    Bitboard occupancy     = occupied();
    Bitboard pinned_pieces = 0;
    // Enemy sliders that would hit our king on an empty board are candidate pinners.
    Bitboard snipers = (rook_attacks(king_square, 0) & (pieces(opponent, ROOK) | pieces(opponent, QUEEN))) |
                       (bishop_attacks(king_square, 0) & (pieces(opponent, BISHOP) | pieces(opponent, QUEEN)));
    while (snipers)
    {
        int      sniper_square = pop_lsb(snipers);
        Bitboard between       = between_bb(king_square, sniper_square) & occupancy;
        // Exactly one piece between the sniper and our king, and it is ours => that piece is pinned.
        if (between && !(between & (between - 1)) && (between & by_color[side]))
        {
            pinned_pieces |= between;
        }
    }
    return pinned_pieces;
}

Bitboard Position::discovered_check_candidates() const
{
    Color    side = stm, opponent = ~side;
    int      enemy_king_square = king_sq(opponent);
    Bitboard occupancy         = occupied();
    Bitboard candidates        = 0;
    // Our sliders that would hit the enemy king on an empty board; a single OWN piece between such a slider
    // and the enemy king is a discovered-check candidate (moving it off the ray delivers check).
    Bitboard snipers = (rook_attacks(enemy_king_square, 0) & (pieces(side, ROOK) | pieces(side, QUEEN))) |
                       (bishop_attacks(enemy_king_square, 0) & (pieces(side, BISHOP) | pieces(side, QUEEN)));
    while (snipers)
    {
        int      sniper_square = pop_lsb(snipers);
        Bitboard between       = between_bb(enemy_king_square, sniper_square) & occupancy;
        if (between && !(between & (between - 1)) && (between & by_color[side]))
        {
            candidates |= between;
        }
    }
    return candidates;
}

bool Position::gives_check_fast(Move move, Bitboard discovered, int enemy_king_square) const
{
    int from = move.from(), to = move.to();

    // Contract: exact only for QUIET non-castle moves (the pruning guards' domain). Anything else reports a
    // conservative "maybe" (true ⇒ never pruned): castling can check with the rook, and captures/promotions —
    // en passant especially, which empties a SECOND square and can open a second discovered ray — need
    // post-capture logic this fast path deliberately does not model.
    if (!move.is_quiet() || move.is_castle())
    {
        return true;
    }

    // Discovered check: the mover leaves the ray between one of our sliders and the enemy king.
    if ((discovered & sq_bb(from)) && !(line_bb(enemy_king_square, from) & sq_bb(to)))
    {
        return true;
    }

    // Direct check from the destination square (quiet move: `to` is empty, the mover leaves `from`).
    Bitboard king_bit  = sq_bb(enemy_king_square);
    Bitboard occupancy = (occupied() ^ sq_bb(from)) | sq_bb(to);
    switch (type_of(board[from]))
    {
    case PAWN:
        return pawn_attacks(stm, to) & king_bit;
    case KNIGHT:
        return knight_attacks(to) & king_bit;
    case BISHOP:
        return bishop_attacks(to, occupancy) & king_bit;
    case ROOK:
        return rook_attacks(to, occupancy) & king_bit;
    case QUEEN:
        return queen_attacks(to, occupancy) & king_bit;
    default:
        return false; // a king never gives direct check
    }
}

bool Position::is_legal_fast(Move move, Bitboard checkers, Bitboard pinned) const
{
    Color side = stm, opponent = ~side;
    int   from = move.from(), to = move.to();
    int   king_square = king_sq(side);

    // Castling is generated fully legal by movegen (king not in/through check, path empty) — always legal.
    if (move.is_castle())
    {
        return true;
    }

    // En passant: removing BOTH pawns (the mover and the captured pawn) can expose our king along a rank or
    // diagonal, so test king safety directly on the post-capture occupancy (no copy-make).
    if (move.is_ep())
    {
        int      captured_square = to + (side == WHITE ? -8 : 8);
        Bitboard after_occupied  = (occupied() ^ sq_bb(from) ^ sq_bb(captured_square)) | sq_bb(to);
        // King must be unattacked after the move: sliders on the post-move occupancy; the captured pawn is
        // dropped from the pawn-attacker set (it is gone), other non-sliders are unaffected by the move.
        if (rook_attacks(king_square, after_occupied) & (pieces(opponent, ROOK) | pieces(opponent, QUEEN)))
        {
            return false;
        }
        if (bishop_attacks(king_square, after_occupied) & (pieces(opponent, BISHOP) | pieces(opponent, QUEEN)))
        {
            return false;
        }
        if (pawn_attacks(side, king_square) & (pieces(opponent, PAWN) ^ sq_bb(captured_square)))
        {
            return false;
        }
        if (knight_attacks(king_square) & pieces(opponent, KNIGHT))
        {
            return false;
        }
        if (king_attacks(king_square) & pieces(opponent, KING))
        {
            return false;
        }
        return true;
    }

    // King move: the destination must be unattacked once the king vacates (so a slider sees through it).
    if (from == king_square)
    {
        return !attackers_to(to, opponent, occupied() ^ sq_bb(king_square));
    }

    // In check: double check leaves only king moves; single check requires capturing the checker or blocking
    // the ray between it and the king.
    if (checkers)
    {
        if (checkers & (checkers - 1))
        {
            return false; // double check, and this is not a king move
        }
        int checker_square = lsb(checkers);
        if (!(sq_bb(to) & (checkers | between_bb(king_square, checker_square))))
        {
            return false;
        }
    }

    // A pinned piece may only move along the pin ray (the line through our king and the piece).
    if ((pinned & sq_bb(from)) && !(line_bb(king_square, from) & sq_bb(to)))
    {
        return false;
    }
    return true;
}

void Position::set_fen(const std::string &fen)
{
    for (int color = 0; color < COLOR_NB; color++)
    {
        by_color[color] = 0;
    }
    for (int piece_type = 0; piece_type < PIECE_TYPE_NB; piece_type++)
    {
        by_type[piece_type] = 0;
    }
    for (int square = 0; square < 64; square++)
    {
        board[square] = NO_PIECE;
    }
    key      = 0;
    pawn_key = 0;
    castling = 0;
    ep_sq    = NO_SQ;
    halfmove = 0;
    fullmove = 1;
    ply      = 0;
    stm      = WHITE;

    std::istringstream stream(fen);
    std::string        board_str, side, castle_str, ep_str;
    stream >> board_str >> side >> castle_str >> ep_str;
    int halfmove_clock = 0, fullmove_number = 1;
    stream >> halfmove_clock >> fullmove_number;

    int file = 0, rank = 7;
    for (char ch : board_str)
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
            PieceType piece_type;
            switch (tolower((unsigned char)ch))
            {
            case 'p':
                piece_type = PAWN;
                break;
            case 'n':
                piece_type = KNIGHT;
                break;
            case 'b':
                piece_type = BISHOP;
                break;
            case 'r':
                piece_type = ROOK;
                break;
            case 'q':
                piece_type = QUEEN;
                break;
            default:
                piece_type = KING;
                break;
            }
            put(color, piece_type, make_square(file, rank));
            file++;
        }
    }

    stm = (side == "b") ? BLACK : WHITE;
    for (char ch : castle_str)
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
    if (ep_str != "-" && ep_str.size() >= 2)
    {
        int ep_square = make_square(ep_str[0] - 'a', ep_str[1] - '1');
        // Keep the ep square only when a pawn of the side to move can actually capture there — matches
        // make_move, so equal positions hash equally regardless of how they were reached.
        if (pawn_attacks(~stm, ep_square) & pieces(stm, PAWN))
        {
            ep_sq = ep_square;
        }
    }
    halfmove = halfmove_clock;
    fullmove = fullmove_number;

    if (stm == BLACK)
    {
        key ^= Zobrist::Side;
    }
    key ^= Zobrist::Castle[castling];
    if (ep_sq != NO_SQ)
    {
        key ^= Zobrist::EpFile[file_of(ep_sq)];
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
        int empty_count = 0;
        for (int file = 0; file < 8; file++)
        {
            Piece piece = board[make_square(file, rank)];
            if (piece == NO_PIECE)
            {
                empty_count++;
                continue;
            }
            if (empty_count)
            {
                result += char('0' + empty_count);
                empty_count = 0;
            }
            const char *names = "PNBRQKpnbrqk";
            result += names[piece];
        }
        if (empty_count)
        {
            result += char('0' + empty_count);
        }
        if (rank)
        {
            result += '/';
        }
    }
    result += stm == WHITE ? " w " : " b ";
    std::string castle_str;
    if (castling & CR_WK)
    {
        castle_str += 'K';
    }
    if (castling & CR_WQ)
    {
        castle_str += 'Q';
    }
    if (castling & CR_BK)
    {
        castle_str += 'k';
    }
    if (castling & CR_BQ)
    {
        castle_str += 'q';
    }
    result += castle_str.empty() ? "-" : castle_str;
    result += ' ';
    result += ep_sq == NO_SQ ? "-" : sq_name(ep_sq);
    result += ' ' + std::to_string(halfmove) + ' ' + std::to_string(fullmove);
    return result;
}
