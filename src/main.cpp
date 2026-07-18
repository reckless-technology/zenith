#include "bitboard.h"
#include "eval.h"
#include "position.h"
#include "search.h"
#include "tt.h"
#include "uci.h"
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv) {
    init_bitboards();
    init_zobrist();
    init_eval();
    init_search();
    TT.resize(64);

    if (argc > 1) {
        if (!strcmp(argv[1], "bench")) {
            run_bench(argc > 2 ? atoi(argv[2]) : 13);
            return 0;
        }
        if (!strcmp(argv[1], "perft")) {
            run_perft_suite();
            return 0;
        }
    }
    uci_loop();
    return 0;
}
