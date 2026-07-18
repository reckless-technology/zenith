#pragma once
// Self-play data generation for NNUE training. Emits one text record per quiet position:
//   fen ; stm_relative_score_cp ; wdl        (wdl in {1.0, 0.5, 0.0} from the side-to-move's POV)
// Zenith generates its own independent dataset — nothing here is shared with any other engine.
//
//   ./zenith datagen <games> <out.txt> [seed] [nodes] [openingPlies]
int run_datagen(int argc, char **argv);
