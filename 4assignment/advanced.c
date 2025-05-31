#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>

// Helper macro to access 2D grid
#define IDX(i, j, size) ((i) * (size) + (j))

// Config struct (if not in gray_scott.h)
typedef struct {
    int n;
    int steps;
    float dt;
    float du;
    float dv;
    float f;
    float k;
} gs_config;

// Simple colormap for visualization
void colormap(float value, unsigned char *r, unsigned char *g, unsigned char *b) {
    float x = fminf(fmaxf(value, 0.0f), 1.0f);
    *r = (unsigned char)(9 * (1 - x) * x * x * x * 255);
    *g = (unsigned char)(15 * (1 - x) * (1 - x) * x * x * 255);
    *b = (unsigned char)(8.5 * (1 - x) * (1 - x) * (1 - x) * x * 255);
}

// Write V grid to PNG (requires stb_image_write)
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void write_png(const char *filename, float *V, int size) {
    unsigned char *image = malloc(size * size * 3);
    float minV = V[0], maxV = V[0];
    for (int i = 1; i < size * size; i++) {
        if (V[i] < minV) minV = V[i];
        if (V[i] > maxV) maxV = V[i];
    }
    float range = (maxV - minV) > 1e-6f ? (maxV - minV) : 1.0f;
    for (int i = 0; i < size * size; i++) {
        float norm = (V[i] - minV) / range;
        unsigned char r, g, b;
        colormap(norm, &r, &g, &b);
        image[3 * i + 0] = r;
        image[3 * i + 1] = g;
        image[3 * i + 2] = b;
    }
    stbi_write_png(filename, size, size, 3, image, size * 3);
    free(image);
}

// Core Gray-Scott solver with MPI row-wise decomposition
double gray_scott2D(const gs_config *config, int rank, int procs) {
    int N = config->n;
    int steps = config->steps;
    float dt = config->dt, Du = config->du, Dv = config->dv, F = config->f, K = config->k;
    int sqrtProcs = (int) round(sqrt(procs));

    if (N % sqrtProcs != 0) {
        if (rank == 0) fprintf(stderr, "Error: grid size (%d) must be multiple of the square root of n_procs (%d)\n", N, sqrtProcs);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int N_local = N / sqrtProcs;
    int N_extended = N_local + 2;

    // 0 1 2
    // 3 4 5
    // 6 7 8
    int rankRow = rank / sqrtProcs;
    int rankCol = rank % sqrtProcs;

    float *U = (float*) malloc(N_extended * N_extended * sizeof(float));
    float *V = (float*) malloc(N_extended * N_extended * sizeof(float));
    float *U_next = (float*) malloc(N_extended * N_extended * sizeof(float));
    float *V_next = (float*) malloc(N_extended * N_extended * sizeof(float));

    // Create derived datatype for the transfer of columns
    MPI_Datatype stridedVector;
    MPI_Type_vector(N_local, 1, N_extended, MPI_FLOAT, &stridedVector);
    MPI_Type_commit(&stridedVector);

    // Initialize U, V
    int h = N/2, r = N/8;
    for (int i = 0; i < N_local; i++) {
        int gi = rankRow * N_local + i;
        for (int j = 0; j < N_local; j++) {
            int gj = rankCol * N_local + j;
            if (gi >= h-r && gi < h+r && gj >= h-r && gj < h+r) {
                U[IDX(i+1, j+1, N_extended)] = 0.75f;
                V[IDX(i+1, j+1, N_extended)] = 0.25f;
            }
            else {
                U[IDX(i+1, j+1, N_extended)] = 1.0f;
                V[IDX(i+1, j+1, N_extended)] = 0.0f;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // Buffers for non-blocking halos
    MPI_Request reqs[16];
    MPI_Status stats[16];

    for (int t = 0; t < steps; t++) {

        // Communication with upper neighbor
        int rankNeighbour = (rank - sqrtProcs + procs) % procs;
        MPI_Isend(&(U[IDX(1, 1, N_extended)]), N_local, MPI_FLOAT, rankNeighbour, 0, MPI_COMM_WORLD, &(reqs[0]));
        MPI_Irecv(&(U[IDX(0, 1, N_extended)]), N_local, MPI_FLOAT, rankNeighbour, 1, MPI_COMM_WORLD, &(reqs[1]));
        MPI_Isend(&(V[IDX(1, 1, N_extended)]), N_local, MPI_FLOAT, rankNeighbour, 2, MPI_COMM_WORLD, &(reqs[2]));
        MPI_Irecv(&(V[IDX(0, 1, N_extended)]), N_local, MPI_FLOAT, rankNeighbour, 3, MPI_COMM_WORLD, &(reqs[3]));
        // Communication with lower neighbor
        rankNeighbour = (rank + sqrtProcs + procs) % procs;
        MPI_Isend(&(U[IDX(N_local, 1, N_extended)]), N_local, MPI_FLOAT, rankNeighbour, 1, MPI_COMM_WORLD, &(reqs[4]));
        MPI_Irecv(&(U[IDX(N_local+1, 1, N_extended)]), N_local, MPI_FLOAT, rankNeighbour, 0, MPI_COMM_WORLD, &(reqs[5]));
        MPI_Isend(&(V[IDX(N_local, 1, N_extended)]), N_local, MPI_FLOAT, rankNeighbour, 3, MPI_COMM_WORLD, &(reqs[6]));
        MPI_Irecv(&(V[IDX(N_local+1, 1, N_extended)]), N_local, MPI_FLOAT, rankNeighbour, 2, MPI_COMM_WORLD, &(reqs[7]));
        // Communication with left neighbor
        rankNeighbour = (rank - 1 + procs) % procs;
        MPI_Isend(&(U[IDX(1, 1, N_extended)]), 1, stridedVector, rankNeighbour, 4, MPI_COMM_WORLD, &(reqs[8]));
        MPI_Irecv(&(U[IDX(1, 0, N_extended)]), 1, stridedVector, rankNeighbour, 5, MPI_COMM_WORLD, &(reqs[9]));
        MPI_Isend(&(V[IDX(1, 1, N_extended)]), 1, stridedVector, rankNeighbour, 6, MPI_COMM_WORLD, &(reqs[10]));
        MPI_Irecv(&(V[IDX(1, 0, N_extended)]), 1, stridedVector, rankNeighbour, 7, MPI_COMM_WORLD, &(reqs[11]));
        // Communication with right neighbor
        rankNeighbour = (rank + 1 + procs) % procs;
        MPI_Isend(&(U[IDX(1, N_local, N_extended)]), 1, stridedVector, rankNeighbour, 5, MPI_COMM_WORLD, &(reqs[12]));
        MPI_Irecv(&(U[IDX(1, N_local+1, N_extended)]), 1, stridedVector, rankNeighbour, 4, MPI_COMM_WORLD, &(reqs[13]));
        MPI_Isend(&(V[IDX(1, N_local, N_extended)]), 1, stridedVector, rankNeighbour, 7, MPI_COMM_WORLD, &(reqs[14]));
        MPI_Irecv(&(V[IDX(1, N_local+1, N_extended)]), 1, stridedVector, rankNeighbour, 6, MPI_COMM_WORLD, &(reqs[15]));

        // Wait for all transfers with both neighbours to finish
        MPI_Waitall(16, reqs, stats);

        // update interior
        for (int i = 1; i < N_extended-1; i++) {
            for (int j = 1; j < N_extended-1; j++) {
                float u = U[IDX(i, j, N_extended)];
                float v = V[IDX(i, j, N_extended)];
                int up = i-1, down = i+1;
                int left = (j-1+N)%N, right = (j+1)%N;
                float lap_u = U[IDX(i-1, j, N_extended)] + U[IDX(i+1, j, N_extended)]
                            + U[IDX(i, j-1, N_extended)] + U[IDX(i, j+1, N_extended)] - 4*u;
                float lap_v = V[IDX(i-1, j, N_extended)] + V[IDX(i+1, j, N_extended)]
                            + V[IDX(i, j-1, N_extended)] + V[IDX(i, j+1, N_extended)] - 4*v;
                float uv2 = u * v * v;
                U_next[IDX(i, j, N_extended)] = u + dt * (-uv2 + F*(1-u) + Du*lap_u);
                V_next[IDX(i, j, N_extended)] = v + dt * ( uv2 - (F+K)*v + Dv*lap_v);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        // swap pointers
        float *tmp;
        tmp = U; U = U_next; U_next = tmp;
        tmp = V; V = V_next; V_next = tmp;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    MPI_Datatype localGrid;
    MPI_Type_vector(N_local, N_local, N_extended, MPI_FLOAT, &localGrid);
    MPI_Type_commit(&localGrid);

    // gather global V in rank 0 for output
    float *V_full = NULL;
    if (rank == 0) V_full = malloc(N * N * sizeof(float));
    MPI_Gather(&V[IDX(1, 1, N_extended)], 1, localGrid,
               V_full, 1, localGrid, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        write_png("output_5000.png", V_full, N);
        free(V_full);
    }

    // compute local avg
    // double local_sum = 0;
    // for (int i = 1; i <= rows; i++) for (int j = 0; j < N; j++) local_sum += V[IDX(i, j, N)];
    // double global_sum;
    // MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    // if (rank == 0) printf("Mean V = %f, Time = %f sec\n", global_sum/(N*(double)N), (t1-t0));

    free(U); free(V); free(U_next); free(V_next);
    return t1 - t0;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    int roundedSqrt = (int) round(sqrt(procs));
    if (roundedSqrt*roundedSqrt != procs) {
        if (rank == 0) fprintf(stderr, "Error: number of workers (%d) must be a perfect square\n", procs);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (argc < 2) {
        if (rank == 0) fprintf(stderr, "Error: usage: %s [grid size]\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int grid_size;
    sscanf(argv[1], "%d", &grid_size);

    if (rank == 0) {
        printf("Gray-Scott MPI: N=%d, procs=%d\n", grid_size, procs);
    }
    gs_config cfg = { grid_size, 5000, 1.0f, 0.16f, 0.08f, 0.06f, 0.062f };
    gray_scott2D(&cfg, rank, procs);

    MPI_Finalize();
    return 0;
}
