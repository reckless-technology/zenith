#pragma once
#include "accumulator.h"
#include "types.h"
#include <string>

// Zobrist keys (filled by init_zobrist()).
namespace Zobrist
{
extern uint64_t Piece[12][64];
extern uint64_t Castle[16];
extern uint64_t EpFile[8];
extern uint64_t Side;
} // namespace Zobrist

void init_zobrist();

// Position is a value type used with copy-make: search copies the parent, then calls make_move on the
// copy. No undo stack — undoing is discarding the copy. All bitboards + the mailbox + the Zobrist key are
// kept in sync incrementally.
class Position
{
  public:
    Bitboard byColor[COLOR_NB]     = {0, 0};
    Bitboard byType[PIECE_TYPE_NB] = {0, 0, 0, 0, 0, 0};
    Piece    board[64];
    Color    stm      = WHITE;
    uint8_t  castling = 0;
    int      epSq     = NO_SQ; // en-passant TARGET square, only set when a capture is actually possible
    int      halfmove = 0;     // 50-move clock (plies)
    int      fullmove = 1;     // full-move number (FEN output only)
    int      ply      = 0;     // plies from the search root (for mate scoring / repetition window)
    uint64_t key      = 0;
    uint64_t pawnKey  = 0; // Zobrist of pawns only, for the eval correction history (search)

    // NNUE accumulator, maintained incrementally in put/remove/move_piece (only when a net is loaded).
    // Copy-make copies it to the child, which make_move then updates by the moved/captured/promoted deltas.
    NnueAccumulator acc;

    Position()
    {
        for (int square = 0; square < 64; square++)
        {
            board[square] = NO_PIECE;
        }
    }

    // --- queries ---
    Bitboard occupied() const
    {
        return byColor[WHITE] | byColor[BLACK];
    }

    Bitboard pieces(Color color, PieceType pieceType) const
    {
        return byColor[color] & byType[pieceType];
    }

    Bitboard pieces(PieceType pieceType) const
    {
        return byType[pieceType];
    }

    int king_sq(Color color) const
    {
        return lsb(pieces(color, KING));
    }

    Piece piece_on(int square) const
    {
        return board[square];
    }

    // Attackers of colour `color` hitting `square`, given `occupancy` (which lets SEE pass an updated board).
    Bitboard attackers_to(int square, Color color, Bitboard occupancy) const;

    bool attacked_by(int square, Color color) const
    {
        return attackers_to(square, color, occupied());
    }

    bool in_check() const
    {
        return attacked_by(king_sq(stm), ~stm);
    }

    bool gives_check(Move move) const; // does move leave the opponent in check?

    // --- mutation ---
    void        set_fen(const std::string &fen);
    std::string fen() const;
    void        make_move(Move move);      // apply move in place (caller copied first)
    void        make_null();               // side-to-move passes (null-move pruning)
    bool        is_legal(Move move) const; // is pseudo-legal move legal (own king not left in check)? [copy-make]

    // Fast legality without copy-make: our pieces pinned to our king, and a legality test given the node's
    // precomputed checkers/pinned. Lets the search prune before paying make_move. is_legal_fast must agree
    // with is_legal on every pseudo-legal move (validated by the `legalcheck` differential test).
    Bitboard pinned_to_king() const;
    bool     is_legal_fast(Move move, Bitboard checkers, Bitboard pinned) const;

    bool has_non_pawn_material(Color color) const
    {
        return byColor[color] & ~(byType[PAWN] | byType[KING]);
    }

  private:
    void put(Color color, PieceType pieceType, int square);
    void remove(int square);
    void move_piece(int from, int to);
};
