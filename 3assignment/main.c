#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>
#include "gray_scott.h"

int main(int argc, char **argv){
    if (argc < 2) {
        fprintf(stderr, "USAGE: %s grid_size\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int grid_size = atoi(argv[1]);

    gs_config config1 = {
        .n = grid_size,
        .steps = 5000,
        .dt = 1.0f,
        .du = 0.16f,
        .dv = 0.08f,
        .f  = 0.06f,
        .k  = 0.062f
    };

    double avgV = gray_scott2D(config1);

    MPI_Finalize();
    return 0;
}
