#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>
#include "gray_scott.h"

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

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
        printf("Benchmark\t   N\tSteps\tMean V\t Time\n");
    gs_config config1 = {.n = 128, .steps = 2000, .dt = 1, .du = 0.04, .dv = 0.02, .f = 0.02, .k = 0.048};
    benchmark(1, rank, config1);
    // For this configuration, the the average concentration of V is 0.11917.
    MPI_Finalize();
    return 0;
}