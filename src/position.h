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
        for (int i = 0; i < 64; i++)
        {
            board[i] = NO_PIECE;
        }
    }

    // --- queries ---
    Bitboard occupied() const
    {
        return byColor[WHITE] | byColor[BLACK];
    }

    Bitboard pieces(Color c, PieceType pt) const
    {
        return byColor[c] & byType[pt];
    }

    Bitboard pieces(PieceType pt) const
    {
        return byType[pt];
    }

    int king_sq(Color c) const
    {
        return lsb(pieces(c, KING));
    }

    Piece piece_on(int sq) const
    {
        return board[sq];
    }

    // Attackers of colour c hitting sq, given occupancy occ (occ lets SEE pass an updated board).
    Bitboard attackers_to(int sq, Color c, Bitboard occ) const;

    bool attacked_by(int sq, Color c) const
    {
        return attackers_to(sq, c, occupied());
    }

    bool in_check() const
    {
        return attacked_by(king_sq(stm), ~stm);
    }

    bool gives_check(Move m) const; // does m leave the opponent in check?

    // --- mutation ---
    void        set_fen(const std::string &fen);
    std::string fen() const;
    void        make_move(Move m);      // apply m in place (caller copied first)
    void        make_null();            // side-to-move passes (null-move pruning)
    bool        is_legal(Move m) const; // is pseudo-legal m legal (own king not left in check)?

    bool has_non_pawn_material(Color c) const
    {
        return byColor[c] & ~(byType[PAWN] | byType[KING]);
    }

  private:
    void put(Color c, PieceType pt, int sq);
    void remove(int sq);
    void move_piece(int from, int to);
};
