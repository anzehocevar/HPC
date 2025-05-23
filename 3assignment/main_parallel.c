#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>
#include "parallel_gray_scott.h"

int benchmark(int case_id, int rank, gs_config config)
{
    double start = omp_get_wtime();
    double meanV = gray_scott2D(config);
    double stop = omp_get_wtime();
    double time = stop - start;
    if (rank == 0)
        printf("%9d\t%4d\t%5d\t%.4f\t%.3f\n", case_id, config.n, config.steps, meanV, time);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 10) {
        fprintf(stderr,
            "USAGE: %s grid_size iterations blockSizeX blockSizeY dt du dv f k\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    int gridSize   = atoi(argv[1]);
    int iterations = atoi(argv[2]);
    int blockSizeX = atoi(argv[3]);
    int blockSizeY = atoi(argv[4]);
    float dt       = atof(argv[5]);
    float du       = atof(argv[6]);
    float dv       = atof(argv[7]);
    float f        = atof(argv[8]);
    float k        = atof(argv[9]);


    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        printf("Benchmark\t   N\tSteps\tMean V\t Time\n");
        printf("Parameters: N=%d, steps=%d, block=%dx%d, dt=%.3f, du=%.3f, dv=%.3f, f=%.3f, k=%.3f\n",
               gridSize, iterations, blockSizeX, blockSizeY, dt, du, dv, f, k);
    }
    // For this configuration, the the average concentration of V is 0.11917.
    // gs_config config1 = {.n = 128, .steps = 2000, .dt = 1, .du = 0.04, .dv = 0.02, .f = 0.02, .k = 0.048};
    // configuration from instructions
    gs_config config = {
        .n = gridSize,
        .steps = iterations,
        .dt = dt,
        .du = du,
        .dv = dv,
        .f = f,
        .k = k
    };
    
    benchmark(1, rank, config);
    MPI_Finalize();
    return 0;
}