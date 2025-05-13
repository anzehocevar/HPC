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

// Must be passed via -DBLOCK_SIZE_... with compiler
#ifndef BLOCK_SIZE_X
#define BLOCK_SIZE_X 16
#endif
#ifndef BLOCK_SIZE_Y
#define BLOCK_SIZE_Y 16
#endif

#define NUM_GPUS 2

// Helper macro to access 2D grid
#define IDX(i, j, size) ((i) * (size) + (j))

__global__ void dummyKernel() {
    // empty warm-up kernel
}

__global__ void gray_scott_kernel(
    int device,
    float *U_half, float *V_half, float *U_half_next, float *V_half_next,
    float *U_middle, float* V_middle,
    int size, int size_half, float dt, float du, float dv, float f, float k) {

    // Get global thread indices
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if ((j >= size) || (i >= size_half))
        return;

    int up    = (i - 1 + size) % size;
    int down  = (i + 1) % size;
    int left  = (j - 1 + size) % size;
    int right = (j + 1) % size;

    int idx       = i * size + j;
    int idx_up    = up * size + j;
    int idx_down  = down * size + j;
    int idx_left  = i * size + left;
    int idx_right = i * size + right;

    float u = U_half[idx];
    float v = V_half[idx];

    float upper_neighbour_U = (device == 1 && i <= 0) ? U_middle[j] : U_half[idx_up];
    float upper_neighbour_V = (device == 1 && i <= 0) ? V_middle[j] : V_half[idx_up];
    float lower_neighbour_U = (device == 0 && i >= blockDim.y - 1) ? U_middle[j] : U_half[idx_down];
    float lower_neighbour_V = (device == 0 && i >= blockDim.y - 1) ? V_middle[j] : V_half[idx_down];

    float lap_u = upper_neighbour_U + lower_neighbour_U + U_half[idx_left] + U_half[idx_right] - 4 * u;
    float lap_v = upper_neighbour_V + lower_neighbour_V + V_half[idx_left] + V_half[idx_right] - 4 * v;

    U_half_next[idx] = u + dt * (du * lap_u - u * v * v + f * (1.0f - u));
    V_half_next[idx] = v + dt * (dv * lap_v + u * v * v - (f + k) * v);
}

__global__ void initUV2D_half(float *U_half, float *V_half, int device, int size, int size_half) {

    // Get global thread indices
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if ((j >= size) || (i >= size_half))
        return;
    int idx = IDX(i, j, size);

    // Set initial values: U=1.0, V=0.0
    U_half[idx] = 1.0f;
    V_half[idx] = 0.0f;

    // Seed a small square in the center
    int h = size / 2;
    int r = size / 8;
    if (j >= h - r && j < h + r) {
        if (
            (device == 0 && i >= h - r) ||
            (device == 1 && i < r)
        ) {
            U_half[idx] = 0.75f;
            V_half[idx] = 0.25f;
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

void enable_p2p() {
    int is_able;

    cudaSetDevice(0);
    cudaDeviceCanAccessPeer(&is_able, 0, 1);
    if (is_able) {
        cudaDeviceEnablePeerAccess(1, 0);
        printf("Enabled P2P: device 0 can access device 1's memory\n");
    }
    else {
        printf("Failed to enable P2P memory transfers\n");
        exit(1);
    }

    cudaSetDevice(1);
    cudaDeviceCanAccessPeer(&is_able, 1, 0);
    if (is_able) {
        cudaDeviceEnablePeerAccess(0, 0);
        printf("Enabled P2P: device 1 can access device 0's memory\n");
    }
    else {
        printf("Failed to enable P2P memory transfers\n");
        exit(1);
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

    enable_p2p();

    if (size % 2 != 0) {
        printf("Grid size must be an even number\n");
        exit(1);
    }
    int size_half = size / 2;

    // Allocate memory
    float *U = (float *)malloc(size * size * sizeof(float));
    float *V = (float *)malloc(size * size * sizeof(float));

    // Allocate device memory
    float *d_U_upper, *d_V_upper, *d_U_upper_next, *d_V_upper_next, *d_U_upper_middle, *d_V_upper_middle;
    float *d_U_lower, *d_V_lower, *d_U_lower_next, *d_V_lower_next, *d_U_lower_middle, *d_V_lower_middle;

    cudaSetDevice(0);
    cudaMalloc((void **)&d_U_upper, size_half * size * sizeof(float));
    cudaMalloc((void **)&d_V_upper, size_half * size * sizeof(float));
    cudaMalloc((void **)&d_U_upper_next, size_half * size * sizeof(float));
    cudaMalloc((void **)&d_V_upper_next, size_half * size * sizeof(float));
    cudaMalloc((void **)&d_U_lower_middle, size * sizeof(float));
    cudaMalloc((void **)&d_V_lower_middle, size * sizeof(float));

    cudaSetDevice(1);
    cudaMalloc((void **)&d_U_lower, size_half * size * sizeof(float));
    cudaMalloc((void **)&d_V_lower, size_half * size * sizeof(float));
    cudaMalloc((void **)&d_U_lower_next, size_half * size * sizeof(float));
    cudaMalloc((void **)&d_V_lower_next, size_half * size * sizeof(float));
    cudaMalloc((void **)&d_U_upper_middle, size * sizeof(float));
    cudaMalloc((void **)&d_V_upper_middle, size * sizeof(float));

    // // Initialize U and V
    // initUV2D(U, V, size);
    // // Copy initial data to device
    // cudaMemcpy(d_U, U, size * size * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_V, V, size * size * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x, (size + blockSize.y - 1) / blockSize.y);
    // Warm-up GPUs
    #pragma omp parallel for num_threads(NUM_GPUS)
    for (int device = 0; device < NUM_GPUS; device++) {
        cudaSetDevice(device);
        dummyKernel<<<1, 1>>>();
        cudaDeviceSynchronize();
    }
    // Start the timer
    double start = MPI_Wtime();
    #pragma omp parallel for num_threads(NUM_GPUS)
    for (int device = 0; device < NUM_GPUS; device++) {
        cudaSetDevice(device);
        initUV2D_half<<<gridSize, blockSize>>>(device ? d_U_lower : d_U_upper, device ? d_V_lower : d_V_upper, device, size, size_half);
        cudaDeviceSynchronize();
    }
    // Main loop
    for (int t = 0; t < iterations; t++) {
        printf("iter: %d\n");
        #pragma omp parallel for num_threads(NUM_GPUS)
        for (int device = 0; device < NUM_GPUS; device++) {
            cudaSetDevice(device);
            float *d_U_half = device ? d_U_lower : d_U_upper;
            float *d_V_half = device ? d_V_lower : d_V_upper;
            float *d_U_half_next = device ? d_U_lower_next : d_U_upper_next;
            float *d_V_half_next = device ? d_V_lower_next : d_V_upper_next;
            float *d_U_middle = device ? d_U_lower_middle : d_U_upper_middle;
            float *d_V_middle = device ? d_V_lower_middle : d_V_upper_middle;
            // Launch kernel
            gray_scott_kernel<<<gridSize, blockSize>>>(device, d_U_half, d_V_half, d_U_half_next, d_V_half_next, d_U_middle, d_V_middle, size, size_half, dt, du, dv, f, k);
            // Copy border elements to other gpu
            if (device == 0) {
                cudaMemcpyPeer(d_U_upper_middle, 1, d_U_upper + (size_half - 1) * size, 0, size * sizeof(float));
                cudaMemcpyPeer(d_V_upper_middle, 1, d_V_upper + (size_half - 1) * size, 0, size * sizeof(float));
            }
            else {
                cudaMemcpyPeer(d_U_lower_middle, 0, d_U_lower, 1, size * sizeof(float));
                cudaMemcpyPeer(d_V_lower_middle, 0, d_V_lower, 1, size * sizeof(float));
            }
            // Synchronize device
            cudaDeviceSynchronize();
            // Swap pointers
            if (device == 0) {
                float *temp_U = d_U_upper;
                d_U_upper = d_U_upper_next;
                d_U_upper_next = temp_U;
                float *temp_V = d_V_upper;
                d_V_upper = d_V_upper_next;
                d_V_upper_next = temp_V;
            }
            else {
                float *temp_U = d_U_lower;
                d_U_lower = d_U_lower_next;
                d_U_lower_next = temp_U;
                float *temp_V = d_V_lower;
                d_V_lower = d_V_lower_next;
                d_V_lower_next = temp_V;
            }
        }
    }
    // Copy result back to host
    cudaSetDevice(0);
    cudaMemcpy(U, d_U_upper, size_half * size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(V, d_V_upper, size_half * size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaSetDevice(1);
    cudaMemcpy(U + size_half, d_U_lower, size_half * size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(V + size_half, d_V_lower, size_half * size * sizeof(float), cudaMemcpyDeviceToHost);
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
    cudaSetDevice(0);
    cudaFree(d_U_upper);
    cudaFree(d_V_upper);
    cudaFree(d_U_upper_next);
    cudaFree(d_V_upper_next);
    cudaFree(d_U_lower_middle);
    cudaFree(d_V_lower_middle);

    cudaSetDevice(1);
    cudaFree(d_U_lower);
    cudaFree(d_V_lower);
    cudaFree(d_U_lower_next);
    cudaFree(d_V_lower_next);
    cudaFree(d_U_upper_middle);
    cudaFree(d_V_upper_middle);

    // Free host memory
    free(U);
    free(V);

    return avgV;
}

