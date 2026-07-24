#pragma once
#include "accumulator.h"
#include "types.h"

// Zobrist keys (filled by init_zobrist()).
extern uint64_t ZobristPiece[12][64];
extern uint64_t ZobristCastle[16];
extern uint64_t ZobristEpFile[8];
extern uint64_t ZobristSide;

void init_zobrist(void);

// Position is a value type used with copy-make: search copies the parent (struct assignment), then calls
// position_make_move on the copy. No undo stack — undoing is discarding the copy. All bitboards + the
// mailbox + the Zobrist key are kept in sync incrementally.
typedef struct Position
{
    Bitboard by_color[COLOR_NB];
    Bitboard by_type[PIECE_TYPE_NB];
    Piece    board[64];
    Color    stm;
    uint8_t  castling;
    int      ep_sq;    // en-passant TARGET square, only set when a capture is actually possible
    int      halfmove; // 50-move clock (plies)
    int      fullmove; // full-move number (FEN output only)
    int      ply;      // plies from the search root (for mate scoring / repetition window)
    uint64_t key;
    uint64_t pawn_key; // Zobrist of pawns only, for the eval correction history (search)

    // NNUE accumulator, maintained incrementally in put/remove/move_piece (only when a net is loaded).
    // Copy-make copies it to the child, which make_move then updates by the moved/captured/promoted deltas.
    NnueAccumulator acc;
} Position;

// Mirrors the C++ default constructor: empty board, White to move, zeroed keys, and — crucially — both
// cached king buckets at 0 so set_fen's incremental put() calls index the net in bounds before the
// authoritative refresh. Call it wherever the C++ declared a fresh `Position pos;`.
static inline void position_init(Position *pos)
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
    pos->stm                = WHITE;
    pos->castling           = 0;
    pos->ep_sq              = NO_SQ;
    pos->halfmove           = 0;
    pos->fullmove           = 1;
    pos->ply                = 0;
    pos->key                = 0;
    pos->pawn_key           = 0;
    pos->acc.king_bucket[0] = 0;
    pos->acc.king_bucket[1] = 0;
}

// --- queries ---
static inline Bitboard position_occupied(const Position *pos)
{
    return pos->by_color[WHITE] | pos->by_color[BLACK];
}

static inline Bitboard position_pieces(const Position *pos, Color color, PieceType piece_type)
{
    return pos->by_color[color] & pos->by_type[piece_type];
}

static inline Bitboard position_pieces_type(const Position *pos, PieceType piece_type)
{
    return pos->by_type[piece_type];
}

static inline int position_king_sq(const Position *pos, Color color)
{
    return lsb(position_pieces(pos, color, KING));
}

static inline Piece position_piece_on(const Position *pos, int square)
{
    return pos->board[square];
}

// Attackers of color `color` hitting `square`, given `occupancy` (which lets SEE pass an updated board).
Bitboard position_attackers_to(const Position *pos, int square, Color color, Bitboard occupancy);

static inline bool position_attacked_by(const Position *pos, int square, Color color)
{
    return position_attackers_to(pos, square, color, position_occupied(pos)) != 0;
}

static inline bool position_in_check(const Position *pos)
{
    return position_attacked_by(pos, position_king_sq(pos, pos->stm), color_flip(pos->stm));
}

// --- mutation ---
// Parse a FEN. Returns false and leaves `pos` unusable if the FEN is malformed or lacks exactly one king
// per side; parsing never writes out of bounds regardless of input. Untrusted-input callers must check it.
bool  position_set_fen(Position *pos, const char *fen);
char *position_fen(const Position *pos, char *buf);      // writes the FEN into buf (>= 128 bytes), returns buf
void  position_make_move(Position *pos, Move move);      // apply move in place (caller copied first)
void  position_make_null(Position *pos);                 // side-to-move passes (null-move pruning)
bool  position_is_legal(const Position *pos, Move move); // is pseudo-legal move legal? [copy-make]

// Fast legality without copy-make: our pieces pinned to our king, and a legality test given the node's
// precomputed checkers/pinned. Lets the search prune before paying make_move. is_legal_fast must agree
// with is_legal on every pseudo-legal move (validated by the `legalcheck` differential test).
Bitboard position_pinned_to_king(const Position *pos);
bool     position_is_legal_fast(const Position *pos, Move move, Bitboard checkers, Bitboard pinned);

// Fast gives-check for QUIET moves, without copy-make: our pieces that would discover check by moving off
// a ray between one of our sliders and the ENEMY king, and the per-move test (direct or discovered check).
// Used as a pruning guard (never prune checking moves) — castling conservatively reports true.
Bitboard position_discovered_check_candidates(const Position *pos);
bool     position_gives_check_fast(const Position *pos, Move move, Bitboard discovered, int enemy_king_square);

static inline bool position_has_non_pawn_material(const Position *pos, Color color)
{
    return (pos->by_color[color] & ~(pos->by_type[PAWN] | pos->by_type[KING])) != 0;
}
