#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>

// Helper macro to access 2D grid
#define IDX(i, j, size) ((i) * (size) + (j))

#ifndef GRID_SIZE
#define GRID_SIZE 256
#endif

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

    if (N % procs != 0) {
        if (rank == 0) fprintf(stderr, "Error: grid size %%d not divisible by procs %%d\n", N, procs);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rows = N / procs;
    int local_rows = rows + 2; // including ghost rows

    float *U = calloc(local_rows * N, sizeof(float));
    float *V = calloc(local_rows * N, sizeof(float));
    float *U_next = calloc(local_rows * N, sizeof(float));
    float *V_next = calloc(local_rows * N, sizeof(float));

    // Initialize U, V on interior rows
    int offset = rank * rows;
    for (int i = 1; i <= rows; i++) {
        int gi = offset + (i - 1);
        for (int j = 0; j < N; j++) {
            U[IDX(i, j, N)] = 1.0f;
            V[IDX(i, j, N)] = 0.0f;
            // seed center
            int r = N / 8;
            if (gi >= N/2 - r && gi < N/2 + r && j >= N/2 - r && j < N/2 + r) {
                U[IDX(i, j, N)] = 0.75f;
                V[IDX(i, j, N)] = 0.25f;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // Buffers for non-blocking halos
    MPI_Request reqs[8];
    MPI_Status stats[8];

    for (int t = 0; t < steps; t++) {
        // Communication with upper neighbor
        MPI_Isend(&U[IDX(1, 0, N)], N, MPI_FLOAT, (rank-1+procs)%procs, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(&U[IDX(0, 0, N)], N, MPI_FLOAT, (rank-1+procs)%procs, 1, MPI_COMM_WORLD, &reqs[1]);
        MPI_Isend(&V[IDX(1, 0, N)], N, MPI_FLOAT, (rank-1+procs)%procs, 2, MPI_COMM_WORLD, &reqs[2]);
        MPI_Irecv(&V[IDX(0, 0, N)], N, MPI_FLOAT, (rank-1+procs)%procs, 3, MPI_COMM_WORLD, &reqs[3]);
        // Communication with lower neighbor
        MPI_Isend(&U[IDX(rows, 0, N)], N, MPI_FLOAT, (rank+1+procs)%procs, 1, MPI_COMM_WORLD, &reqs[4]);
        MPI_Irecv(&U[IDX(rows+1, 0, N)], N, MPI_FLOAT, (rank+1+procs)%procs, 0, MPI_COMM_WORLD, &reqs[5]);
        MPI_Isend(&V[IDX(rows, 0, N)], N, MPI_FLOAT, (rank+1+procs)%procs, 3, MPI_COMM_WORLD, &reqs[6]);
        MPI_Irecv(&V[IDX(rows+1, 0, N)], N, MPI_FLOAT, (rank+1+procs)%procs, 2, MPI_COMM_WORLD, &reqs[7]);

        // Wait for all transfers with both neighbours to finish
        MPI_Waitall(8, reqs, stats);

        // update interior
        for (int i = 1; i <= rows; i++) {
            for (int j = 0; j < N; j++) {
                float u = U[IDX(i, j, N)];
                float v = V[IDX(i, j, N)];
                int up = i-1, down = i+1;
                int left = (j-1+N)%N, right = (j+1)%N;
                float lap_u = U[IDX(up, j, N)] + U[IDX(down, j, N)]
                            + U[IDX(i, left, N)] + U[IDX(i, right, N)] - 4*u;
                float lap_v = V[IDX(up, j, N)] + V[IDX(down, j, N)]
                            + V[IDX(i, left, N)] + V[IDX(i, right, N)] - 4*v;
                float uv2 = u * v * v;
                U_next[IDX(i, j, N)] = u + dt * (-uv2 + F*(1-u) + Du*lap_u);
                V_next[IDX(i, j, N)] = v + dt * ( uv2 - (F+K)*v + Dv*lap_v);
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

    // gather global V in rank 0 for output
    float *V_full = NULL;
    if (rank == 0) V_full = malloc(N * N * sizeof(float));
    MPI_Gather(&V[IDX(1, 0, N)], rows * N, MPI_FLOAT,
               V_full, rows * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        write_png("output_5000.png", V_full, N);
        free(V_full);
    }

    // compute local avg
    double local_sum = 0;
    for (int i = 1; i <= rows; i++) for (int j = 0; j < N; j++) local_sum += V[IDX(i, j, N)];
    double global_sum;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Mean V = %f, Time = %f sec\n", global_sum/(N*(double)N), (t1-t0));

    free(U); free(V); free(U_next); free(V_next);
    return t1 - t0;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    if (rank == 0) {
        printf("Gray-Scott MPI: N=%d, procs=%d\n", GRID_SIZE, procs);
    }
    gs_config cfg = { GRID_SIZE, 5000, 1.0f, 0.16f, 0.08f, 0.06f, 0.062f };
    gray_scott2D(&cfg, rank, procs);

    MPI_Finalize();
    return 0;
}
