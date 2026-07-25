// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief Board mutation and queries: Zobrist init, make_move, FEN I/O, and the copy-free legality oracles.
 */
#include "position.h"
#include "bitboard.h"
#include "nnue.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

uint64_t ZobristPiece[12][64];
uint64_t ZobristCastle[16];
uint64_t ZobristEpFile[8];
uint64_t ZobristSide;

static uint64_t zobrist_next(uint64_t *state)
{
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return *state * 0x2545F4914F6CDD1DULL;
}

void init_zobrist(void)
{
    uint64_t state = 0x9E3779B97F4A7C15ULL;
    for (int piece = 0; piece < 12; piece++)
    {
        for (int square = 0; square < 64; square++)
        {
            ZobristPiece[piece][square] = zobrist_next(&state);
        }
    }
    for (int castle_rights = 0; castle_rights < 16; castle_rights++)
    {
        ZobristCastle[castle_rights] = zobrist_next(&state);
    }
    for (int file = 0; file < 8; file++)
    {
        ZobristEpFile[file] = zobrist_next(&state);
    }
    ZobristSide = zobrist_next(&state);
}

// CastleMask[sq]: rights to KEEP when a piece leaves or arrives on sq (AND-ed into castling).
// Everything defaults to CR_ALL; the six rook/king home squares clear their rights (a1 e1 h1, a8 e8 h8).
static const uint8_t CastleMask[64] = {
    (uint8_t)~CR_WQ, CR_ALL, CR_ALL,          CR_ALL, (uint8_t) ~(CR_WK | CR_WQ),
    CR_ALL,          CR_ALL, (uint8_t)~CR_WK, // rank 1
    CR_ALL,          CR_ALL, CR_ALL,          CR_ALL, CR_ALL,
    CR_ALL,          CR_ALL, CR_ALL, // rank 2
    CR_ALL,          CR_ALL, CR_ALL,          CR_ALL, CR_ALL,
    CR_ALL,          CR_ALL, CR_ALL, // rank 3
    CR_ALL,          CR_ALL, CR_ALL,          CR_ALL, CR_ALL,
    CR_ALL,          CR_ALL, CR_ALL, // rank 4
    CR_ALL,          CR_ALL, CR_ALL,          CR_ALL, CR_ALL,
    CR_ALL,          CR_ALL, CR_ALL, // rank 5
    CR_ALL,          CR_ALL, CR_ALL,          CR_ALL, CR_ALL,
    CR_ALL,          CR_ALL, CR_ALL, // rank 6
    CR_ALL,          CR_ALL, CR_ALL,          CR_ALL, CR_ALL,
    CR_ALL,          CR_ALL, CR_ALL, // rank 7
    (uint8_t)~CR_BQ, CR_ALL, CR_ALL,          CR_ALL, (uint8_t) ~(CR_BK | CR_BQ),
    CR_ALL,          CR_ALL, (uint8_t)~CR_BK, // rank 8
};

/** @brief Place a piece and update every representation: bitboards, mailbox, Zobrist keys, NNUE accumulator. */
static void put(Position *pos, const Color color, const PieceType piece_type, const int square)
{
    const Bitboard square_bit = sq_bb(square);
    pos->by_color[color] |= square_bit;
    pos->by_type[piece_type] |= square_bit;
    const Piece piece  = make_piece(color, piece_type);
    pos->board[square] = piece;
    pos->key ^= ZobristPiece[piece][square];
    if (piece_type == PAWN)
    {
        pos->pawn_key ^= ZobristPiece[piece][square];
    }
    if (nnue_is_loaded())
    {
        nnue_add_feature(&pos->acc, color, piece_type, square);
    }
}

/** @brief Remove the piece on @p square and update every representation (inverse of put). */
static void remove_piece(Position *pos, const int square)
{
    const Piece    piece      = pos->board[square];
    const Bitboard square_bit = sq_bb(square);
    pos->by_color[color_of(piece)] ^= square_bit;
    pos->by_type[type_of(piece)] ^= square_bit;
    pos->key ^= ZobristPiece[piece][square];
    if (type_of(piece) == PAWN)
    {
        pos->pawn_key ^= ZobristPiece[piece][square];
    }
    pos->board[square] = NO_PIECE;
    if (nnue_is_loaded())
    {
        nnue_remove_feature(&pos->acc, color_of(piece), type_of(piece), square);
    }
}

/** @brief Move the piece @p from -> @p to (no capture) and update every representation. */
static void move_piece(Position *pos, const int from, const int to)
{
    const Piece    piece    = pos->board[from];
    const Bitboard from_bit = sq_bb(from), to_bit = sq_bb(to);
    const Bitboard from_to_bits = from_bit | to_bit;
    pos->by_color[color_of(piece)] ^= from_to_bits;
    pos->by_type[type_of(piece)] ^= from_to_bits;
    pos->key ^= ZobristPiece[piece][from] ^ ZobristPiece[piece][to];
    if (type_of(piece) == PAWN)
    {
        pos->pawn_key ^= ZobristPiece[piece][from] ^ ZobristPiece[piece][to];
    }
    pos->board[to]   = piece;
    pos->board[from] = NO_PIECE;
    if (nnue_is_loaded())
    {
        nnue_move_feature(&pos->acc, color_of(piece), type_of(piece), from, to);
    }
}

Bitboard position_attackers_to(const Position *pos, const int square, const Color color, const Bitboard occupancy)
{
    Bitboard attackers = 0;
    attackers |= pawn_attacks(color_flip(color), square) & position_pieces(pos, color, PAWN);
    attackers |= knight_attacks(square) & position_pieces(pos, color, KNIGHT);
    attackers |= king_attacks(square) & position_pieces(pos, color, KING);
    attackers |=
        bishop_attacks(square, occupancy) & (position_pieces(pos, color, BISHOP) | position_pieces(pos, color, QUEEN));
    attackers |=
        rook_attacks(square, occupancy) & (position_pieces(pos, color, ROOK) | position_pieces(pos, color, QUEEN));
    return attackers;
}

/** @brief Apply @p move to @p pos in place (copy-make: the caller copied @p pos first — there is no unmake). */
void position_make_move(Position *pos, const Move move)
{
    const Color     side = pos->stm, opponent = color_flip(pos->stm);
    const int       from = move_from(move), to = move_to(move);
    const PieceType piece_type = type_of(pos->board[from]);

    if (pos->ep_sq != NO_SQ)
    {
        pos->key ^= ZobristEpFile[file_of(pos->ep_sq)];
        pos->ep_sq = NO_SQ;
    }
    pos->halfmove++;

    if (move_is_capture(move))
    {
        pos->halfmove = 0;
        if (move_is_ep(move))
        {
            remove_piece(pos, to + (side == WHITE ? -8 : 8)); // captured pawn sits behind the ep target
        }
        else
        {
            remove_piece(pos, to);
        }
    }

    if (move_is_castle(move))
    {
        const int rank = rank_of(from);
        if (move_flag(move) == FLAG_KCASTLE)
        {
            move_piece(pos, make_square(7, rank), make_square(5, rank));
        }
        else
        {
            move_piece(pos, make_square(0, rank), make_square(3, rank));
        }
    }

    move_piece(pos, from, to);

    if (piece_type == PAWN)
    {
        pos->halfmove = 0;
        if (move_is_promo(move))
        {
            remove_piece(pos, to);
            put(pos, side, move_promo_pt(move), to);
        }
        else if (move_is_double(move))
        {
            const int ep_square = (from + to) / 2;
            if (pawn_attacks(side, ep_square) & position_pieces(pos, opponent, PAWN))
            {
                pos->ep_sq = ep_square;
                pos->key ^= ZobristEpFile[file_of(ep_square)];
            }
        }
    }

    // King-input buckets: a king move (including castling) can change the moving side's king bucket, which
    // shifts that whole perspective's feature block — refresh it. The primitives above updated both
    // perspectives incrementally with the pre-move buckets; refresh_perspective discards the stale own half.
    if (piece_type == KING && nnue_is_loaded())
    {
        nnue_update_king_bucket(&pos->acc, pos, side);
    }

    const uint8_t old_castling = pos->castling;
    pos->castling &= CastleMask[from] & CastleMask[to];
    if (pos->castling != old_castling)
    {
        pos->key ^= ZobristCastle[old_castling] ^ ZobristCastle[pos->castling];
    }

    if (side == BLACK)
    {
        pos->fullmove++;
    }
    pos->stm = opponent;
    pos->key ^= ZobristSide;
    pos->ply++;
}

void position_make_null(Position *pos)
{
    if (pos->ep_sq != NO_SQ)
    {
        pos->key ^= ZobristEpFile[file_of(pos->ep_sq)];
        pos->ep_sq = NO_SQ;
    }
    pos->stm = color_flip(pos->stm);
    pos->key ^= ZobristSide;
    pos->halfmove++;
    pos->ply++;
}

bool position_is_legal(const Position *pos, const Move move)
{
    const Color side = pos->stm;
    Position    copy = *pos;
    position_make_move(&copy, move);
    return !position_is_attacked_by(&copy, position_king_sq(&copy, side), color_flip(side));
}

Bitboard position_pinned_to_king(const Position *pos)
{
    const Color    side = pos->stm, opponent = color_flip(side);
    const int      king_square   = position_king_sq(pos, side);
    const Bitboard occupancy     = position_occupied(pos);
    Bitboard       pinned_pieces = 0;
    // Enemy sliders that would hit our king on an empty board are candidate pinners.
    Bitboard snipers = (rook_attacks(king_square, 0) &
                        (position_pieces(pos, opponent, ROOK) | position_pieces(pos, opponent, QUEEN))) |
                       (bishop_attacks(king_square, 0) &
                        (position_pieces(pos, opponent, BISHOP) | position_pieces(pos, opponent, QUEEN)));
    while (snipers)
    {
        const int      sniper_square = pop_lsb(&snipers);
        const Bitboard between       = between_bb(king_square, sniper_square) & occupancy;
        // Exactly one piece between the sniper and our king, and it is ours => that piece is pinned.
        if (between && !(between & (between - 1)) && (between & pos->by_color[side]))
        {
            pinned_pieces |= between;
        }
    }
    return pinned_pieces;
}

Bitboard position_discovered_check_candidates(const Position *pos)
{
    const Color    side = pos->stm, opponent = color_flip(side);
    const int      enemy_king_square = position_king_sq(pos, opponent);
    const Bitboard occupancy         = position_occupied(pos);
    Bitboard       candidates        = 0;
    // Our sliders that would hit the enemy king on an empty board; a single OWN piece between such a slider
    // and the enemy king is a discovered-check candidate (moving it off the ray delivers check).
    Bitboard snipers =
        (rook_attacks(enemy_king_square, 0) & (position_pieces(pos, side, ROOK) | position_pieces(pos, side, QUEEN))) |
        (bishop_attacks(enemy_king_square, 0) &
         (position_pieces(pos, side, BISHOP) | position_pieces(pos, side, QUEEN)));
    while (snipers)
    {
        const int      sniper_square = pop_lsb(&snipers);
        const Bitboard between       = between_bb(enemy_king_square, sniper_square) & occupancy;
        if (between && !(between & (between - 1)) && (between & pos->by_color[side]))
        {
            candidates |= between;
        }
    }
    return candidates;
}

/** @brief Copy-free gives-check test for a QUIET @p move (direct or discovered check). */
bool position_gives_check_fast(const Position *pos, const Move move, const Bitboard discovered,
                               const int enemy_king_square)
{
    const int from = move_from(move), to = move_to(move);

    // Contract: exact only for QUIET non-castle moves (the pruning guards' domain). Anything else reports a
    // conservative "maybe" (true ⇒ never pruned): castling can check with the rook, and captures/promotions —
    // en passant especially, which empties a SECOND square and can open a second discovered ray — need
    // post-capture logic this fast path deliberately does not model.
    if (!move_is_quiet(move) || move_is_castle(move))
    {
        return true;
    }

    // Discovered check: the mover leaves the ray between one of our sliders and the enemy king.
    if ((discovered & sq_bb(from)) && !(line_bb(enemy_king_square, from) & sq_bb(to)))
    {
        return true;
    }

    // Direct check from the destination square (quiet move: `to` is empty, the mover leaves `from`).
    const Bitboard king_bit  = sq_bb(enemy_king_square);
    const Bitboard occupancy = (position_occupied(pos) ^ sq_bb(from)) | sq_bb(to);
    switch (type_of(pos->board[from]))
    {
    case PAWN:
        return (pawn_attacks(pos->stm, to) & king_bit) != 0;
    case KNIGHT:
        return (knight_attacks(to) & king_bit) != 0;
    case BISHOP:
        return (bishop_attacks(to, occupancy) & king_bit) != 0;
    case ROOK:
        return (rook_attacks(to, occupancy) & king_bit) != 0;
    case QUEEN:
        return (queen_attacks(to, occupancy) & king_bit) != 0;
    default:
        return false; // a king never gives direct check
    }
}

/** @brief Copy-free legality test for a pseudo-legal @p move, given the node's @p checkers and @p pinned. */
bool position_is_legal_fast(const Position *pos, const Move move, const Bitboard checkers, const Bitboard pinned)
{
    const Color side = pos->stm, opponent = color_flip(side);
    const int   from = move_from(move), to = move_to(move);
    const int   king_square = position_king_sq(pos, side);

    // Castling is generated fully legal by movegen (king not in/through check, path empty) — always legal.
    if (move_is_castle(move))
    {
        return true;
    }

    // En passant: removing BOTH pawns (the mover and the captured pawn) can expose our king along a rank or
    // diagonal, so test king safety directly on the post-capture occupancy (no copy-make).
    if (move_is_ep(move))
    {
        const int      captured_square = to + (side == WHITE ? -8 : 8);
        const Bitboard after_occupied  = (position_occupied(pos) ^ sq_bb(from) ^ sq_bb(captured_square)) | sq_bb(to);
        // King must be unattacked after the move: sliders on the post-move occupancy; the captured pawn is
        // dropped from the pawn-attacker set (it is gone), other non-sliders are unaffected by the move.
        if (rook_attacks(king_square, after_occupied) &
            (position_pieces(pos, opponent, ROOK) | position_pieces(pos, opponent, QUEEN)))
        {
            return false;
        }
        if (bishop_attacks(king_square, after_occupied) &
            (position_pieces(pos, opponent, BISHOP) | position_pieces(pos, opponent, QUEEN)))
        {
            return false;
        }
        if (pawn_attacks(side, king_square) & (position_pieces(pos, opponent, PAWN) ^ sq_bb(captured_square)))
        {
            return false;
        }
        if (knight_attacks(king_square) & position_pieces(pos, opponent, KNIGHT))
        {
            return false;
        }
        if (king_attacks(king_square) & position_pieces(pos, opponent, KING))
        {
            return false;
        }
        return true;
    }

    // King move: the destination must be unattacked once the king vacates (so a slider sees through it).
    if (from == king_square)
    {
        return !position_attackers_to(pos, to, opponent, position_occupied(pos) ^ sq_bb(king_square));
    }

    // In check: double check leaves only king moves; single check requires capturing the checker or blocking
    // the ray between it and the king.
    if (checkers)
    {
        if (checkers & (checkers - 1))
        {
            return false; // double check, and this is not a king move
        }
        const int checker_square = lsb(checkers);
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

/** @brief Parse @p fen into @p pos, hardened against malformed input. @return false if malformed/kingless. */
bool position_set_fen(Position *pos, const char *fen)
{
    for (int color = 0; color < COLOR_NB; color++)
    {
        pos->by_color[color] = 0;
    }
    for (int piece_type = 0; piece_type < PIECE_TYPE_NB; piece_type++)
    {
        pos->by_type[piece_type] = 0;
    }
    for (int square = 0; square < 64; square++)
    {
        pos->board[square] = NO_PIECE;
    }
    pos->key      = 0;
    pos->pawn_key = 0;
    pos->castling = 0;
    pos->ep_sq    = NO_SQ;
    pos->halfmove = 0;
    pos->fullmove = 1;
    pos->ply      = 0;
    pos->stm      = WHITE;

    // Tokenize a local copy on whitespace; missing trailing fields read as "".
    char fen_copy[512];
    snprintf(fen_copy, sizeof fen_copy, "%s", fen);
    char             *save_ptr     = NULL;
    const char *const separators   = " \t\r\n";
    const char       *board_str    = strtok_r(fen_copy, separators, &save_ptr);
    const char       *side         = strtok_r(NULL, separators, &save_ptr);
    const char       *castle_str   = strtok_r(NULL, separators, &save_ptr);
    const char       *ep_str       = strtok_r(NULL, separators, &save_ptr);
    const char *const halfmove_str = strtok_r(NULL, separators, &save_ptr);
    const char *const fullmove_str = strtok_r(NULL, separators, &save_ptr);
    if (board_str == NULL)
    {
        board_str = "";
    }
    if (side == NULL)
    {
        side = "";
    }
    if (castle_str == NULL)
    {
        castle_str = "";
    }
    if (ep_str == NULL)
    {
        ep_str = "";
    }
    // Lenient integer fields: a failed/absent halfmove read stores 0 and stops later parsing, so a 4-field
    // FEN (no clocks) yields halfmove 0, fullmove 1.
    int   halfmove_clock = 0, fullmove_number = 1;
    char *end_ptr = NULL;
    if (halfmove_str != NULL)
    {
        const long parsed    = strtol(halfmove_str, &end_ptr, 10);
        const bool is_parsed = end_ptr != halfmove_str;
        halfmove_clock       = is_parsed ? (int)parsed : 0;
        if (is_parsed && *end_ptr == '\0' && fullmove_str != NULL)
        {
            const long full = strtol(fullmove_str, &end_ptr, 10);
            fullmove_number = end_ptr != fullmove_str ? (int)full : 0;
        }
        else
        {
            fullmove_number = is_parsed ? 0 : 1; // trailing junk / failed first read, as the stream behaves
        }
    }

    // Piece placement. Every square index is bounds-checked before put(): a malformed board field (over-long
    // rank, too many '/', stray digits) can drive file/rank out of [0,8), and put() writes board[square] +
    // indexes Zobrist[..][square] + shifts 1<<square, so an unchecked square is a memory-safety hole reachable
    // from a single `position fen` line. Out-of-range placements are skipped rather than written.
    int  file = 0, rank = 7;
    bool is_malformed = false;
    for (const char *cursor = board_str; *cursor; cursor++)
    {
        const char ch = *cursor;
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
            const Color color = isupper((unsigned char)ch) ? WHITE : BLACK;
            PieceType   piece_type;
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
            case 'k':
                piece_type = KING;
                break;
            default:
                is_malformed = true; // unknown piece letter
                continue;
            }
            if (file >= 0 && file < 8 && rank >= 0 && rank < 8)
            {
                put(pos, color, piece_type, make_square(file, rank));
            }
            else
            {
                is_malformed = true; // placement outside the board — skip, don't write OOB
            }
            file++;
        }
    }

    pos->stm = strcmp(side, "b") == 0 ? BLACK : WHITE;
    for (const char *cursor = castle_str; *cursor; cursor++)
    {
        switch (*cursor)
        {
        case 'K':
            pos->castling |= CR_WK;
            break;
        case 'Q':
            pos->castling |= CR_WQ;
            break;
        case 'k':
            pos->castling |= CR_BK;
            break;
        case 'q':
            pos->castling |= CR_BQ;
            break;
        default:
            break;
        }
    }
    if (strcmp(ep_str, "-") != 0 && strlen(ep_str) >= 2)
    {
        const int ep_file = ep_str[0] - 'a', ep_rank = ep_str[1] - '1';
        // Validate the coordinates before make_square: an out-of-range ep field (e.g. "z9") would otherwise
        // index pawn_attacks[..][ep_square] out of bounds. Keep the ep square only when a pawn of the side to
        // move can actually capture there — matches make_move, so equal positions hash equally.
        if (ep_file >= 0 && ep_file < 8 && ep_rank >= 0 && ep_rank < 8)
        {
            const int ep_square = make_square(ep_file, ep_rank);
            if (pawn_attacks(color_flip(pos->stm), ep_square) & position_pieces(pos, pos->stm, PAWN))
            {
                pos->ep_sq = ep_square;
            }
        }
    }
    pos->halfmove = halfmove_clock;
    pos->fullmove = fullmove_number;

    if (pos->stm == BLACK)
    {
        pos->key ^= ZobristSide;
    }
    pos->key ^= ZobristCastle[pos->castling];
    if (pos->ep_sq != NO_SQ)
    {
        pos->key ^= ZobristEpFile[file_of(pos->ep_sq)];
    }
    // A legal position has exactly one king per side. Reject anything else: a kingless side would make
    // position_king_sq() do lsb(0) (ctz of zero is UB) and then read the attack tables out of bounds during
    // search. Callers handling untrusted input (UCI `position fen`, datagen openings) must honour `false`.
    bool is_valid = !is_malformed && popcount(position_pieces(pos, WHITE, KING)) == 1 &&
                    popcount(position_pieces(pos, BLACK, KING)) == 1;

    // Authoritative accumulator rebuild (put() updated it incrementally from an uninitialised state above).
    if (nnue_is_loaded())
    {
        nnue_refresh(&pos->acc, pos);
    }
    return is_valid;
}

char *position_fen(const Position *pos, char *buf)
{
    char *out = buf;
    for (int rank = 7; rank >= 0; rank--)
    {
        int empty_count = 0;
        for (int file = 0; file < 8; file++)
        {
            const Piece piece = pos->board[make_square(file, rank)];
            if (piece == NO_PIECE)
            {
                empty_count++;
                continue;
            }
            if (empty_count)
            {
                *out++      = (char)('0' + empty_count);
                empty_count = 0;
            }
            const char *const names = "PNBRQKpnbrqk";
            *out++                  = names[piece];
        }
        if (empty_count)
        {
            *out++ = (char)('0' + empty_count);
        }
        if (rank)
        {
            *out++ = '/';
        }
    }
    *out++ = ' ';
    *out++ = pos->stm == WHITE ? 'w' : 'b';
    *out++ = ' ';
    if (pos->castling == 0)
    {
        *out++ = '-';
    }
    else
    {
        if (pos->castling & CR_WK)
        {
            *out++ = 'K';
        }
        if (pos->castling & CR_WQ)
        {
            *out++ = 'Q';
        }
        if (pos->castling & CR_BK)
        {
            *out++ = 'k';
        }
        if (pos->castling & CR_BQ)
        {
            *out++ = 'q';
        }
    }
    *out++ = ' ';
    char square_name[3];
    sq_name(pos->ep_sq, square_name);
    out += sprintf(out, "%s %d %d", square_name, pos->halfmove, pos->fullmove);
    (void)out;
    return buf;
}
