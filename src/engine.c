// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief Engine construction and destruction (see engine.h for the aggregate itself).
 */
#include "engine.h"
#include "bitboard.h"
#include "nnue.h"
#include <stdlib.h>

Engine *engine_new(void)
{
    init_bitboards(); // idempotent: the attack tables are process-wide and filled on first construction

    Engine *const engine = calloc(1, sizeof(Engine));
    if (engine == NULL)
    {
        return NULL;
    }
    search_shared_init(&engine->search);
    eval_cache_init(&engine->eval_cache); // a failed cache allocation degrades to uncached eval, not an error
    tt_resize(&engine->tt, ENGINE_DEFAULT_HASH_MB);
    if (engine->tt.table == NULL)
    {
        engine_delete(engine); // a searchless engine is useless — fail construction, leaking nothing
        return NULL;
    }
    return engine;
}

void engine_delete(Engine *engine)
{
    if (engine == NULL)
    {
        return;
    }
    tt_free(&engine->tt);
    eval_cache_free(&engine->eval_cache);
    nnue_free(engine->net);
    free(engine);
}
