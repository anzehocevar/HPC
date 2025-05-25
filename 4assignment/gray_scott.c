#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "gray_scott.h"

#ifndef GRID_SIZE
#define GRID_SIZE 128
#endif

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


double gray_scott2D(gs_config config){
    // Initialize vars from .h
    int size = config.n;
    int iterations = config.steps;
    float dt = config.dt;
    float du = config.du;
    float dv = config.dv;
    float f = config.f;
    float k = config.k;

    // Allocate memory
    float *U = (float *)malloc(size * size * sizeof(float));
    float *V = (float *)malloc(size * size * sizeof(float));
    float *U_next = (float *)malloc(size * size * sizeof(float));
    float *V_next = (float *)malloc(size * size * sizeof(float));

    // Start stopwatch
    double start = omp_get_wtime();
    
    // Initialize U and V
    initUV2D(U, V, size);
    
    // Main loop
    for(int it = 0;it < iterations; it++){
        // Update U and V using the Gray-Scott model
        #pragma omp parallel for collapse(2)
        for(int i = 0;i < size; i++){
            for(int j = 0;j < size;j++){
                // Get the indices of the neighbors
                int up = (i - 1 + size) % size;
                int down = (i + 1) % size;
                int left = (j - 1 + size) % size;
                int right = (j + 1) % size;

                // Compute the Laplacian
                float laplacian_U = U[IDX(up, j, size)] + U[IDX(down, j, size)] +
                                    U[IDX(i, left, size)] + U[IDX(i, right, size)] -
                                    4 * U[IDX(i, j, size)];

                float laplacian_V = V[IDX(up, j, size)] + V[IDX(down, j, size)] +
                                    V[IDX(i, left, size)] + V[IDX(i, right, size)] -
                                    4 * V[IDX(i, j, size)];

                // Update U and V
                U_next[IDX(i, j, size)] = U[IDX(i, j, size)] +
                                                dt * (du * laplacian_U - U[IDX(i, j, size)] * V[IDX(i, j, size)] * V[IDX(i, j, size)] +
                                                f * (1 - U[IDX(i, j, size)]));

                V_next[IDX(i, j, size)] = V[IDX(i, j, size)] +
                                                dt * (dv * laplacian_V + U[IDX(i,j,size)] * V[IDX(i,j,size)] * V[IDX(i,j,size)] -
                                                (f + k) * V[IDX(i,j,size)]);
            }
        }

        // Swap pointers
        float *temp = U;
        U = U_next;
        U_next = temp;

        temp = V;
        V = V_next;
        V_next = temp;
    }


    double end = omp_get_wtime();
    double time = end - start;
    printf("Elapsed time: %.3f seconds\n", time);

    // return average concentartion of V
    double avgV = 0.0;
    for (int i = 0; i < size * size; i++) {
        avgV += V[i];
    }
    avgV /= (size * size);

    // Write output to file
    write_png("output.png", V, size);

    // Cleanup
    free(U);
    free(V);
    free(U_next);
    free(V_next);

    return avgV;
}


int benchmark(int case_id, gs_config config){
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
