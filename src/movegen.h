#pragma once
#include "position.h"

struct MoveList {
    Move moves[256];
    int cnt = 0;
    void add(Move m) { moves[cnt++] = m; }
    Move* begin() { return moves; }
    Move* end() { return moves + cnt; }
    const Move* begin() const { return moves; }
    const Move* end() const { return moves + cnt; }
    int size() const { return cnt; }
    Move& operator[](int i) { return moves[i]; }
};

// Fully legal moves. noisyOnly restricts to captures + promotions (for quiescence).
void generate_legal(const Position& pos, MoveList& list, bool noisyOnly = false);
