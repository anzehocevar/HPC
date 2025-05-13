#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "gray_scott.h"

int benchmark(int case_id, gs_config config)
{
    double start = omp_get_wtime();
    double meanV = gray_scott2D(config);
    double stop = omp_get_wtime();
    double time = stop - start;
    printf("%9d\t%4d\t%5d\t%.4f\t%.3f\n", case_id, config.n, config.steps, meanV, time);
    return 0;
}

int main(int argc, char **argv)
{
    printf("Benchmark\t   N\tSteps\tMean V\t Time\n");
    // For this configuration, the the average concentration of V is 0.11917.
    // gs_config config1 = {.n = 128, .steps = 2000, .dt = 1, .du = 0.04, .dv = 0.02, .f = 0.02, .k = 0.048};
    // configuration from instructions
    gs_config config1 = {.n = GRID_SIZE, .steps = 5000, .dt = 1, .du = 0.16, .dv = 0.08, .f = 0.06, .k = 0.062};
    benchmark(1, config1);
    return 0;
}
