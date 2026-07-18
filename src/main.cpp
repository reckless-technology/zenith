#include "bitboard.h"
#include "datagen.h"
#include "eval.h"
#include "nnue.h"
#include "position.h"
#include "search.h"
#include "tt.h"
#include "uci.h"
#include <cstdlib>
#include <cstring>

int main(int argc, char **argv)
{
    init_bitboards();
    init_zobrist();
    init_eval();
    init_search();
    TT.resize(64);

    if (argc > 1)
    {
        if (!strcmp(argv[1], "bench"))
        {
            run_bench(argc > 2 ? atoi(argv[2]) : 13);
            return 0;
        }
        if (!strcmp(argv[1], "perft"))
        {
            run_perft_suite();
            return 0;
        }
        if (!strcmp(argv[1], "datagen"))
        {
            return run_datagen(argc - 1, argv + 1);
        }
        if (!strcmp(argv[1], "nnueeval"))
        {
            if (argc < 3)
            {
                fprintf(stderr, "usage: %s nnueeval <net.nnue>   (reads FENs from stdin)\n", argv[0]);
                return 1;
            }
            return nnue::eval_fens_from_stdin(argv[2]);
        }
    }
    uci_loop();
    return 0;
}
