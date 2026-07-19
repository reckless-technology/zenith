#pragma once
#include "position.h"

struct MoveList
{
    Move moves[256];
    int  cnt = 0;

    void add(Move m)
    {
        moves[cnt++] = m;
    }

    Move *begin()
    {
        return moves;
    }

    Move *end()
    {
        return moves + cnt;
    }

    const Move *begin() const
    {
        return moves;
    }

    const Move *end() const
    {
        return moves + cnt;
    }

    int size() const
    {
        return cnt;
    }

    Move &operator[](int i)
    {
        return moves[i];
    }
};

// Fully legal moves. noisyOnly restricts to captures + promotions (for quiescence).
void generate_legal(const Position &pos, MoveList &list, bool noisyOnly = false);

// Pseudo-legal moves (castling already fully legal). The search makes each move once and skips those that
// leave the mover's king in check — avoiding generate_legal's extra copy-make per move.
void generate_pseudo(const Position &pos, MoveList &list, bool noisyOnly = false);
