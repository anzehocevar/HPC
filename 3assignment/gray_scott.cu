#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include "gray_scott.h"

// Helper macro to access 2D grid
#define IDX(i, j, size) ((i) * (size) + (j))

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"


// visualize
void colormap(float value, unsigned char *r, unsigned char *g, unsigned char *b) {
    float x = fminf(fmaxf(value, 0.0f), 1.0f);
    *r = (unsigned char)(9*(1-x)*x*x*x*255);
    *g = (unsigned char)(15*(1-x)*(1-x)*x*x*255);
    *b = (unsigned char)(8.5*(1-x)*(1-x)*(1-x)*x*255);
}

void write_png(const char *filename, float *V, int size) {
    unsigned char *image = (unsigned char *)malloc(size * size * 3);

    float minV = V[0], maxV = V[0];
    for (int i = 1; i < size * size; i++) {
        if (V[i] < minV) minV = V[i];
        if (V[i] > maxV) maxV = V[i];
    }
    float range = maxV - minV;
    if (range < 1e-6f) range = 1.0f;

    for (int i = 0; i < size * size; i++) {
        float norm = (V[i] - minV) / range;
        unsigned char r, g, b;
        colormap(norm, &r, &g, &b);
        image[i * 3 + 0] = r;
        image[i * 3 + 1] = g;
        image[i * 3 + 2] = b;
    }

    stbi_write_png_compression_level = 9;
    stbi_write_png(filename, size, size, 3, image, size * 3);
    free(image);
}



// Reference function for initialization of U and V
void initUV2D(float *U, float *V, int size) {
    // Set initial values: U=1.0, V=0.0
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            U[IDX(i, j, size)] = 1.0f;
            V[IDX(i, j, size)] = 0.0f;
        }
    }

    // Seed a small square in the center
    int r = size / 8;
    for (int i = size / 2 - r; i < size / 2 + r; i++) {
        for (int j = size / 2 - r; j < size / 2 + r; j++) {
            U[IDX(i, j, size)] = 0.75f;
            V[IDX(i, j, size)] = 0.25f;
        }
    }
}


double gray_scott2D(gs_config config) {
    int size = config.n;
    int iterations = config.steps;
    float dt = config.dt;
    float du = config.du;
    float dv = config.dv;
    float f = config.f;
    float k = config.k;

    // Allocate
    float *U = (float *)malloc(size * size * sizeof(float));
    float *V = (float *)malloc(size * size * sizeof(float));
    float *U_next = (float *)malloc(size * size * sizeof(float));
    float *V_next = (float *)malloc(size * size * sizeof(float));

    // Init time measurement
    double init_start = MPI_Wtime();
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int idx = i * size + j;
            U[idx] = 1.0f;
            V[idx] = 0.0f;
        }
    }

    int r = size / 8;
    for (int i = size / 2 - r; i < size / 2 + r; i++) {
        for (int j = size / 2 - r; j < size / 2 + r; j++) {
            int idx = i * size + j;
            U[idx] = 0.75f;
            V[idx] = 0.25f;
        }
    }
    double init_end = MPI_Wtime();

    // Simulation loop timing
    double sim_start = MPI_Wtime();
    for (int t = 0; t < iterations; t++) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                int idx = i * size + j;
                int up    = (i - 1 + size) % size;
                int down  = (i + 1) % size;
                int left  = (j - 1 + size) % size;
                int right = (j + 1) % size;

                float lap_u = U[up * size + j] + U[down * size + j] +
                              U[i * size + left] + U[i * size + right] - 4 * U[idx];

                float lap_v = V[up * size + j] + V[down * size + j] +
                              V[i * size + left] + V[i * size + right] - 4 * V[idx];

                U_next[idx] = U[idx] + dt * (du * lap_u - U[idx] * V[idx] * V[idx] + f * (1.0f - U[idx]));
                V_next[idx] = V[idx] + dt * (dv * lap_v + U[idx] * V[idx] * V[idx] - (f + k) * V[idx]);
            }
        }

        float *tmp_u = U; U = U_next; U_next = tmp_u;
        float *tmp_v = V; V = V_next; V_next = tmp_v;
    }
    double sim_end = MPI_Wtime();

    // Compute avg V
    double avgV = 0.0;
    for (int i = 0; i < size * size; i++) avgV += V[i];
    avgV /= (size * size);

    // Print in grep-friendly form
    printf("Init_time:%.6f\n", init_end - init_start);
    printf("Compute_time:%.6f\n", sim_end - sim_start);
    printf("Average concentration of V: %.6f\n", avgV);

    // Save the final state of V to a PNG file
    write_png("output_seq.png", V, size);

    // Clean up
    free(U);
    free(V);
    free(U_next);
    free(V_next);

    return avgV;
}


