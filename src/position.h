// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief The board representation (a copy-make value type), Zobrist keys, and the position oracles.
 */
#pragma once
#include "accumulator.h"
#include "types.h"

#include <stddef.h>
#include <string.h>

/**
 * @brief The board, a value type used with copy-make.
 *
 * Search copies the parent (struct assignment), then calls position_make_move on the copy. No undo stack —
 * undoing is discarding the copy. All bitboards + the mailbox + the Zobrist key are kept in sync
 * incrementally.
 */
typedef struct Position
{
    Bitboard colors[NUM_COLORS]; ///< Occupancy per color.

    Bitboard pieces[NUM_PIECES]; ///< Occupancy per piece type (both colors), indexed by Piece. The NO_PIECE slot (index
    ///< zero) holds the occupied-squares bitboard — all piece bitboards OR-ed — maintained
    ///< incrementally alongside the rest.

    uint64_t key;      ///< Incremental Zobrist key of the whole position.
    uint64_t pawn_key; ///< Zobrist of pawns only, for the eval correction history (search).

    /// Pieces giving check to the side to move — recomputed once per make_move/make_null/set_fen, so the
    /// search and movegen read it instead of recomputing attackers-to-king. 0 iff the side to move is not in check.
    Bitboard checkers;

    /// Mailbox of 1-byte Piece codes (NO_PIECE=0 .. KING=6). Color is not stored here — read it from
    /// the `colors` bitboards via position_color_on. 1-byte entries keep the copy-make struct small.
    uint8_t board[NUM_SQUARES];

    // Scalars, widest-first. The semantic fields carry their enum types (Color/Square) for readability;
    // those are int-sized but land in the struct's existing tail padding before the 32-aligned accumulator,
    // so sizeof(Position) is unchanged. The move counters stay narrow: they are bounded by the draw rules
    // (half_move <= ~150 under the 75-move rule; full_move fits any game in 16 bits), and an out-of-range FEN
    // counter truncates harmlessly (half_move only feeds the >=100 draw test + repetition window; full_move is
    // FEN-output only).
    Color    color_to_move;             ///< side to move
    Square   ep_square;                 ///< en-passant TARGET square (0..63), or NO_SQUARE when no ep is possible
    Square   king_location[NUM_COLORS]; ///< king square per color (0 if kingless), maintained by add_piece/move_piece
    uint16_t full_move;                 ///< full-move number (FEN output only)
    uint8_t  half_move;                 ///< 50-move clock (plies); resets on a pawn move or capture
    uint8_t  castling_rights;           ///< castling-rights bitmask (CR_*)

    /// NNUE accumulator, maintained incrementally in add_piece/remove_piece/move_piece (only when a net is loaded).
    /// Copy-make copies it to the child, which make_move then updates by the moved/captured/promoted deltas.
    NnueAccumulator accumulator;
} Position;

// The accumulator must stay the final member for position_copy_for_make's prefix copy to cover every other
// field. sizeof(Position) is the accumulator's offset plus its size — no trailing padding to miss.
_Static_assert(offsetof(Position, accumulator) + sizeof(NnueAccumulator) == sizeof(Position),
               "the NNUE accumulator must be Position's last member (position_copy_for_make copies the prefix)");

/**
 * @brief Copy @p src to @p dst ahead of a position_make_move — the copy half of copy-make.
 *
 * A net-bound position gets the plain struct copy (the accumulator must reach the child, which make_move then
 * updates incrementally). A position with no net — perft, the legality/SEE gates; see position_init — never
 * reads the accumulator, and it is ~95% of sizeof(Position), so the copy skips it: only the prefix is copied
 * and `values` is left indeterminate (never read, because the net binding stays NULL and every accumulator
 * update is guarded on it). That alone makes the eval-free walks ~2.4x faster.
 */
static inline void position_copy_for_make(Position *dst, const Position *src)
{
    if (src->accumulator.net != NULL)
    {
        *dst = *src;
        return;
    }
    memcpy(dst, src, offsetof(Position, accumulator));
    dst->accumulator.net   = NULL;
    dst->accumulator.cache = NULL;
}

/**
 * @brief Initialise @p pos to an empty board bound to @p net: White to move, zeroed keys, both cached king
 * buckets at 0.
 *
 * The zeroed king buckets are crucial so set_fen's incremental add_piece() calls index the net in bounds before
 * the authoritative refresh. @p net may be NULL only for positions that are never evaluated (perft, the
 * legality/SEE/book gates — the accumulator updates become no-ops); anything that reaches evaluate() must be
 * bound to the engine's net. Call it before using any freshly-declared `Position`.
 */
static inline void position_init(Position *pos, const NnueNetwork *net)
{
    memset(pos, 0, sizeof(Position));
    pos->accumulator.net = net;
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

/// @}

/// @name Mutation
/// @{
/**
 * @brief Parse a FEN into @p pos.
 * @return false (leaving @p pos unusable) if the FEN is malformed, lacks exactly one king per side, or has
 * more than 16 pieces of either color (impossible in chess; the cap is also what makes MAX_MOVES provably
 * sufficient for unchecked move emission — see movegen.h).
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
/** @brief Whether a pseudo-legal @p move is legal, tested via copy-make — the slow reference oracle (ground
 *  truth for the `legalcheck` gate). Prefer position_is_move_legal, the copy-free path movegen and the search use. */
bool position_is_move_legal_slow(const Position *pos, Move move);
/**
 * @brief position_is_move_legal_slow, additionally handing back the position it built in @p child_out.
 *
 * The oracle's copy-make child is also the ground truth for other post-move properties (notably
 * `child_out->checkers`, the gives-check truth). Taking it from here lets `legalcheck` check both oracles per
 * move off ONE copy-make instead of three — the copy dominates the gate's runtime.
 */
bool position_is_move_legal_slow_child(const Position *pos, Move move, Position *child_out);

/**
 * @brief Our pieces pinned to our own king.
 *
 * Part of the fast, copy-free legality path: with this and the node's checkers, position_is_move_legal can
 * prune before paying make_move. It must agree with position_is_move_legal_slow on every pseudo-legal move
 * (validated by the `legalcheck` differential test).
 */
Bitboard position_pinned_to_king(const Position *pos);
/**
 * @brief Copy-free legality test for a pseudo-legal @p move — the primary legality filter (movegen's
 * generate_legal and the search both use it; position_is_move_legal_slow is the copy-make oracle it is
 * validated against). Reads the position's cached checkers; passing the once-per-node @p pinned makes it
 * O(1) per move — no make_move.
 * @param pos the position.
 * @param move the pseudo-legal move to test.
 * @param pinned our pieces pinned to our king (see position_pinned_to_king).
 */
bool position_is_move_legal(const Position *pos, Move move, Bitboard pinned);

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

/**
 * @brief Whether @p pos is drawn by the 50-move rule, insufficient material (K/K+minor vs K/K+minor), or
 * 2-fold repetition against @p history_keys (the Zobrist keys of the positions played before @p pos, oldest
 * first — the search passes its game+tree history, datagen its game history). The repetition scan steps by
 * 2 (same side to move) back to the last irreversible move.
 */
bool position_is_draw(const Position *pos, const uint64_t *history_keys, int history_count);

/** @brief Whether @p color has any non-pawn, non-king material (used to gate null-move pruning). */
static inline bool position_has_non_pawn_material(const Position *pos, Color color)
{
    return (pos->colors[color] & ~(pos->pieces[PAWN] | pos->pieces[KING])) != 0;
}

/// @}
