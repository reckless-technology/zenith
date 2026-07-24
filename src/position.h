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
    Bitboard by_color[COLOR_NB]     = {0, 0};
    Bitboard by_type[PIECE_TYPE_NB] = {0, 0, 0, 0, 0, 0};
    Piece    board[64];
    Color    stm      = WHITE;
    uint8_t  castling = 0;
    int      ep_sq    = NO_SQ; // en-passant TARGET square, only set when a capture is actually possible
    int      halfmove = 0;     // 50-move clock (plies)
    int      fullmove = 1;     // full-move number (FEN output only)
    int      ply      = 0;     // plies from the search root (for mate scoring / repetition window)
    uint64_t key      = 0;
    uint64_t pawn_key = 0; // Zobrist of pawns only, for the eval correction history (search)

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
        return by_color[WHITE] | by_color[BLACK];
    }

    Bitboard pieces(Color color, PieceType piece_type) const
    {
        return by_color[color] & by_type[piece_type];
    }

    Bitboard pieces(PieceType piece_type) const
    {
        return by_type[piece_type];
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

    // Fast gives-check for QUIET moves, without copy-make: our pieces that would discover check by moving off
    // a ray between one of our sliders and the ENEMY king, and the per-move test (direct or discovered check).
    // Used as a pruning guard (never prune checking moves) — castling conservatively reports true.
    Bitboard discovered_check_candidates() const;
    bool     gives_check_fast(Move move, Bitboard discovered, int enemy_king_square) const;

    bool has_non_pawn_material(Color color) const
    {
        return by_color[color] & ~(by_type[PAWN] | by_type[KING]);
    }

  private:
    void put(Color color, PieceType piece_type, int square);
    void remove(int square);
    void move_piece(int from, int to);
};
