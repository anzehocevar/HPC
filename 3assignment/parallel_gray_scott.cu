#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include "parallel_gray_scott.h"

#include <cuda_runtime.h>
#include <cuda.h>
#include "helper_cuda.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

// Helper macro to access 2D grid
#define IDX(i, j, size) ((i) * (size) + (j))

__global__ void dummyKernel() {
    // empty warm-up kernel
}

__global__ void gray_scott_kernel_1(
    float *U, float *V, float *U_next, float *V_next,
    int size, float dt, float du, float dv, float f, float k) {

    __shared__ float s_U[1 + BLOCK_SIZE_Y + 1][1 + BLOCK_SIZE_X + 1];
    __shared__ float s_V[1 + BLOCK_SIZE_Y + 1][1 + BLOCK_SIZE_X + 1];

    // Get global thread indices
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size || j >= size) return;

    int up    = (i - 1 + size) % size;
    int down  = (i + 1) % size;
    int left  = (j - 1 + size) % size;
    int right = (j + 1) % size;

    int idx       = i * size + j;
    int idx_up    = up * size + j;
    int idx_down  = down * size + j;
    int idx_left  = i * size + left;
    int idx_right = i * size + right;

    int s_y = threadIdx.y + 1;
    int s_x = threadIdx.x + 1;

    s_U[s_y][s_x] = U[idx];
    s_V[s_y][s_x] = V[idx];

    if (threadIdx.x == 0) {
        s_U[s_y][0] = U[idx_left];
        s_V[s_y][0] = V[idx_left];
    }
    if (threadIdx.y == 0) {
        s_U[0][s_x] = U[idx_up];
        s_V[0][s_x] = V[idx_up];
    }
    if (threadIdx.x >= blockDim.x - 1) {
        s_U[s_y][blockDim.x + 1] = U[idx_right];
        s_V[s_y][blockDim.x + 1] = V[idx_right];
    }
    if (threadIdx.y >= blockDim.y - 1) {
        s_U[blockDim.y + 1][s_x] = U[idx_down];
        s_V[blockDim.y + 1][s_x] = V[idx_down];
    }

    float u = s_U[s_y][s_x];
    float v = s_V[s_y][s_x];

    __syncthreads();

    float lap_u = s_U[s_y-1][s_x] + s_U[s_y+1][s_x] + s_U[s_y][s_x-1] + s_U[s_y][s_x+1] - 4 * u;
    float lap_v = s_V[s_y-1][s_x] + s_V[s_y+1][s_x] + s_V[s_y][s_x-1] + s_V[s_y][s_x+1] - 4 * v;

    float uv2 = u * v * v;

    U_next[idx] = u + dt * (du * lap_u - uv2 + f * (1.0f - u));
    V_next[idx] = v + dt * (dv * lap_v + uv2 - (f + k) * v);

}

__global__ void gray_scott_kernel(
    float *U, float *V, float *U_next, float *V_next,
    int size, float dt, float du, float dv, float f, float k) {

    // Get global thread indices
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size || j >= size) return;

    int up    = (i - 1 + size) % size;
    int down  = (i + 1) % size;
    int left  = (j - 1 + size) % size;
    int right = (j + 1) % size;

    int idx       = i * size + j;
    int idx_up    = up * size + j;
    int idx_down  = down * size + j;
    int idx_left  = i * size + left;
    int idx_right = i * size + right;

    float u = U[idx];
    float v = V[idx];

    float lap_u = U[idx_up] + U[idx_down] + U[idx_left] + U[idx_right] - 4 * u;
    float lap_v = V[idx_up] + V[idx_down] + V[idx_left] + V[idx_right] - 4 * v;

    U_next[idx] = u + dt * (du * lap_u - u * v * v + f * (1.0f - u));
    V_next[idx] = v + dt * (dv * lap_v + u * v * v - (f + k) * v);
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

    
    // Initialize U and V
    initUV2D(U, V, size);

    // Allocate device memory
    float *d_U, *d_V, *d_U_next, *d_V_next;
    cudaMalloc((void **)&d_U, size * size * sizeof(float));
    cudaMalloc((void **)&d_V, size * size * sizeof(float));

    cudaMalloc((void **)&d_U_next, size * size * sizeof(float));
    cudaMalloc((void **)&d_V_next, size * size * sizeof(float));

    // Copy initial data to device
    cudaMemcpy(d_U, U, size * size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, size * size * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x, (size + blockSize.y - 1) / blockSize.y);
    // Warm-up GPU
    dummyKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    // Start the timer
    double start = MPI_Wtime();
    // Main loop
    for (int t = 0; t < iterations; t++) {
        // Launch kernel
        gray_scott_kernel_1<<<gridSize, blockSize>>>(d_U, d_V, d_U_next, d_V_next, size, dt, du, dv, f, k);
        // Synchronize device
        cudaDeviceSynchronize();
        // Swap pointers
        float *temp_U = d_U;
        d_U = d_U_next;
        d_U_next = temp_U;

        float *temp_V = d_V;
        d_V = d_V_next;
        d_V_next = temp_V;
    }
    // Copy result back to host
    cudaMemcpy(U, d_U, size * size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(V, d_V, size * size * sizeof(float), cudaMemcpyDeviceToHost);
    // Stop the timer
    double end = MPI_Wtime();
    double elapsed = end - start;
    printf("Elapsed time: %f seconds\n", elapsed);
    // Print average concentration of U
    double avgU = 0.0;
    for (int i = 0; i < size * size; i++) {
        avgU += U[i];
    }
    avgU /= (size * size);
    printf("Average concentration of U: %f\n", avgU);
    // Print average concentration of V
    double avgV = 0.0;
    for (int i = 0; i < size * size; i++) {
        avgV += V[i];
    }
    avgV /= (size * size);
    printf("Average concentration of V: %f\n", avgV);

    // Write output to file
    write_png("output.png", V, size);
    
    // Free device memory
    cudaFree(d_U);
    cudaFree(d_V);
    cudaFree(d_U_next);
    cudaFree(d_V_next);

    // Free host memory
    free(U);
    free(V);
    free(U_next);
    free(V_next);


    return avgV;
}


