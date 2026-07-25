// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief The board representation (a copy-make value type), Zobrist keys, and the position oracles.
 */
#pragma once
#include "accumulator.h"
#include "types.h"

#include <string.h>

/// @name Zobrist keys (filled by init_zobrist()).
/// @{
extern uint64_t ZobristPiece[NUM_COLORS][NUM_PIECES]
                            [NUM_SQUARES]; ///< per (color, piece type, square); [*][NO_PIECE][*] unused
extern uint64_t ZobristCastle[16];         ///< per castling-rights mask
extern uint64_t ZobristEp[NUM_SQUARES];    ///< per en-passant target square (non-zero only on ranks 3 and 6)
extern uint64_t ZobristSide;               ///< XOR-ed in when Black is to move
/// @}

/** @brief Fill the Zobrist key tables. Call once at startup (after init_bitboards). */
void init_zobrist(void);

/**
 * @brief The board, a value type used with copy-make.
 *
 * Search copies the parent (struct assignment), then calls position_make_move on the copy. No undo stack —
 * undoing is discarding the copy. All bitboards + the mailbox + the Zobrist key are kept in sync
 * incrementally.
 */
typedef struct Position
{
    Bitboard colors[NUM_COLORS]; ///< occupancy per color
    /// Occupancy per piece type (both colors), indexed by Piece. The NO_PIECE slot (index 0) holds the
    /// occupied-squares bitboard — all piece bitboards OR-ed — maintained incrementally alongside the rest.
    Bitboard pieces[NUM_PIECES];
    uint64_t key;      ///< incremental Zobrist key of the whole position
    uint64_t pawn_key; ///< Zobrist of pawns only, for the eval correction history (search)
    /// Mailbox of 1-byte Piece codes (NO_PIECE=0 .. KING=6). Color is not stored here — read it from
    /// the `colors` bitboards via position_color_on. 1-byte entries keep the copy-make struct small.
    uint8_t board[NUM_SQUARES];
    // Scalars packed widest-first. halfmove/fullmove stay 32-bit (a FEN may specify large values, and a
    // narrow type would silently truncate); ep_square (0..NUM_SQUARES), ply (0..MAX_PLY), stm and castling are small.
    int32_t halfmove;        ///< 50-move clock (plies)
    int32_t fullmove;        ///< full-move number (FEN output only)
    int16_t ep_square;       ///< en-passant TARGET square, only set when a capture is actually possible
    int16_t ply;             ///< plies from the search root (for mate scoring / repetition window)
    uint8_t color_to_move;   ///< side to move (Color; stored narrow — values are 0/1)
    uint8_t castling_rights; ///< castling-rights bitmask (CR_*)

    /// NNUE accumulator, maintained incrementally in put/remove/move_piece (only when a net is loaded).
    /// Copy-make copies it to the child, which make_move then updates by the moved/captured/promoted deltas.
    NnueAccumulator accumulator;
} Position;

/**
 * @brief Initialise @p pos to an empty board: White to move, zeroed keys, both cached king buckets at 0.
 *
 * The zeroed king buckets are crucial so set_fen's incremental put() calls index the net in bounds before
 * the authoritative refresh. Call it before using any freshly-declared `Position`.
 */
static inline void position_init(Position *pos)
{
    memset(pos, 0, sizeof(Position));
}

/// @name Queries
/// @{
/** @brief All occupied squares (both colors) — the incrementally-maintained pieces[NO_PIECE] slot. */
static inline Bitboard position_occupied(const Position *pos)
{
    return pos->pieces[NO_PIECE];
}

/** @brief Squares holding @p color pieces of @p piece. */
static inline Bitboard position_pieces(const Position *pos, Color color, Piece piece)
{
    return pos->colors[color] & pos->pieces[piece];
}

/** @brief Squares holding @p piece pieces of either color. */
static inline Bitboard position_pieces_type(const Position *pos, Piece piece)
{
    return pos->pieces[piece];
}

/** @brief Square of @p color's king (undefined if that side has no king). */
static inline int position_king_sq(const Position *pos, Color color)
{
    return lsb(position_pieces(pos, color, KING));
}

/** @brief The piece type on @p square, or NO_PIECE if empty. */
static inline Piece position_piece_on(const Position *pos, int square)
{
    return (Piece)pos->board[square];
}

/** @brief The color of the piece on @p square (undefined for empty squares — check occupancy first). */
static inline Color position_color_on(const Position *pos, int square)
{
    return (Color)((pos->colors[BLACK] >> square) & 1);
}

/**
 * @brief Attackers of color @p color hitting @p square.
 * @param pos the position.
 * @param square the target square.
 * @param color the attacking side.
 * @param occupancy the occupancy to test against (lets SEE pass an updated board).
 */
Bitboard position_attackers_to(const Position *pos, int square, Color color, Bitboard occupancy);

/** @brief Whether @p square is attacked by any @p color piece (on the current occupancy). */
static inline bool position_is_attacked_by(const Position *pos, int square, Color color)
{
    return position_attackers_to(pos, square, color, position_occupied(pos)) != 0;
}

/** @brief Whether the side to move is in check. */
static inline bool position_is_in_check(const Position *pos)
{
    return position_is_attacked_by(pos, position_king_sq(pos, pos->color_to_move), enemy_of(pos->color_to_move));
}

/// @}

/// @name Mutation
/// @{
/**
 * @brief Parse a FEN into @p pos.
 * @return false (leaving @p pos unusable) if the FEN is malformed or lacks exactly one king per side.
 *
 * Parsing never writes out of bounds regardless of input. Untrusted-input callers must check the result.
 */
bool position_set_fen(Position *pos, const char *fen);
/** @brief Write the FEN of @p pos into @p buf (>= 128 bytes). @return @p buf. */
char *position_fen(const Position *pos, char *buf);
/** @brief Apply @p move to @p pos in place (the caller must have copied @p pos first — there is no unmake). */
void position_make_move(Position *pos, Move move);
/** @brief The side to move passes (for null-move pruning). */
void position_make_null(Position *pos);
/** @brief Whether a pseudo-legal @p move is legal, tested via copy-make. */
bool position_is_legal(const Position *pos, Move move);

/**
 * @brief Our pieces pinned to our own king.
 *
 * Part of the fast, copy-free legality path: with this and the node's checkers, position_is_legal_fast can
 * prune before paying make_move. It must agree with position_is_legal on every pseudo-legal move
 * (validated by the `legalcheck` differential test).
 */
Bitboard position_pinned_to_king(const Position *pos);
/**
 * @brief Copy-free legality test for a pseudo-legal @p move.
 * @param pos the position.
 * @param move the pseudo-legal move to test.
 * @param checkers pieces giving check to the side to move (precomputed once per node).
 * @param pinned our pieces pinned to our king (see position_pinned_to_king).
 */
bool position_is_legal_fast(const Position *pos, Move move, Bitboard checkers, Bitboard pinned);

/**
 * @brief Our pieces that could discover check by moving off a ray between one of our sliders and the enemy king.
 *
 * Feeds the fast gives-check test used as a pruning guard (never prune a checking move).
 */
Bitboard position_discovered_check_candidates(const Position *pos);
/**
 * @brief Copy-free gives-check test for a QUIET @p move (direct or discovered check).
 * @param pos the position.
 * @param move the quiet move to test.
 * @param discovered candidates from position_discovered_check_candidates.
 * @param enemy_king_square the square of the king being checked.
 *
 * Castling conservatively reports true (it is only a pruning guard).
 */
bool position_gives_check_fast(const Position *pos, Move move, Bitboard discovered, int enemy_king_square);

/** @brief Whether @p color has any non-pawn, non-king material (used to gate null-move pruning). */
static inline bool position_has_non_pawn_material(const Position *pos, Color color)
{
    return (pos->colors[color] & ~(pos->pieces[PAWN] | pos->pieces[KING])) != 0;
}

/// @}
