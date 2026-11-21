/*
 * pdl-bench — SPDX-License-Identifier: MIT
 *
 * pdl_train — train a CNN with a sequential or distributed strategy and
 * write a result JSON (docs/RESULTS_SCHEMA.md).
 *
 *   build/bin/pdl_train --strategy sequential --epochs 3 --output out.json
 *   mpirun -np 4 build/bin/pdl_train --strategy sync --global-batch 256 --output out.json
 */
#include <string.h>

#include "pdl/common.h"
#include "pdl/comm.h"
#include "pdl/train.h"

int main(int argc, char **argv) {
    int rank = 0, world = 1;
    pdl_mpi_init(&argc, &argv, &rank, &world);

    pdl_train_config cfg;
    pdl_train_config_defaults(&cfg);
    int rc = pdl_train_config_parse(&cfg, argc, argv);
    if (rc != 0) {
        if (rank == 0) {
            if (rc > 0)
                pdl_train_config_print_help(stdout, argv[0]);
            else
                fprintf(stderr, "Run with --help for the list of options.\n");
        }
        pdl_mpi_finalize();
        return rc > 0 ? 0 : 2;
    }

    pdl_trainer trainer;
    pdl_trainer_init(&trainer, &cfg, argc, argv);
    pdl_trainer_run(&trainer);
    long failures = trainer.sync_failures;
    pdl_trainer_finalize(&trainer);
    pdl_mpi_finalize();
    return failures ? 3 : 0;
}
