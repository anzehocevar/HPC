
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include "helper_cuda.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define COLOR_CHANNELS 0
#define LUMINANCE_LEVELS 256

// Global device-managed memory for histogram, cumulative distribution function (cdf), and lookup table (lut)
__device__ __managed__ int d_histogram[LUMINANCE_LEVELS];
__device__ __managed__ int d_cdf[LUMINANCE_LEVELS];
__device__ __managed__ unsigned char d_lut[LUMINANCE_LEVELS];

// Compute local histograms in shared memory -> reduce to global histogram
__global__ void computeHistogramShared(const unsigned char *imageIn, int width, int height, int cpp) {
    __shared__ unsigned int localHist[LUMINANCE_LEVELS];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Initialize shared histogram
    for (int i = tid; i < LUMINANCE_LEVELS; i += blockDim.x * blockDim.y)
        localHist[i] = 0;
    __syncthreads();

    // Get global indexes
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    int gidy = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if within image bounds
    if (gidx < width && gidy < height) {
        // Each thread computes luminance and updates local histogram
        float r = imageIn[(gidy * width + gidx) * cpp + 0];
        float g = imageIn[(gidy * width + gidx) * cpp + 1];
        float b = imageIn[(gidy * width + gidx) * cpp + 2];
        unsigned char Y = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
        atomicAdd(&localHist[Y], 1);
    }
    __syncthreads();

    // Merge shared histogram to global histogram
    for (int i = tid; i < LUMINANCE_LEVELS; i += blockDim.x * blockDim.y)
        atomicAdd(&d_histogram[i], localHist[i]);
}

// Parallel Blelloch scan to compute cumulative histogram (prefix sum)
__global__ void blellochScan(int *cdf, int n) {
    __shared__ int temp[LUMINANCE_LEVELS * 2];
    int tid = threadIdx.x;

    int offset = 1;
    for (int i = 0; i < n; i++)
        temp[2 * i] = temp[2 * i + 1] = 0;
    __syncthreads();

    if (tid < n) temp[tid] = d_histogram[tid];
    __syncthreads();

    // Up-sweep phase
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d)
            temp[offset * (2 * tid + 2) - 1] += temp[offset * (2 * tid + 1) - 1];
        offset *= 2;
    }

    // Clear last element
    if (tid == 0) temp[n - 1] = 0;

    // Down-sweep phase
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    if (tid < n) cdf[tid] = temp[tid];
}

// Compute lookup table for new luminance values based on CDF
__global__ void computeLUT(int *cdf, int size, int totalPixels) {
    int tid = threadIdx.x;
    __shared__ int minCdf;

    // Determine the minimum non-zero CDF value (for contrast stretching)
    if (tid == 0) {
        for (int i = 0; i < size; i++) {
            if (cdf[i] > 0) {
                minCdf = cdf[i];
                break;
            }
        }
    }
    __syncthreads();

    // Compute output luminance via histogram equalization formula
    if (tid < size) {
        d_lut[tid] = (unsigned char)(roundf(((float)(cdf[tid] - minCdf) / (totalPixels - minCdf)) * (LUMINANCE_LEVELS - 1)));
    }
}

// Apply new luminance values using the LUT and convert back to RGB
__global__ void applyEqualization(unsigned char *imageIn, unsigned char *imageOut, int width, int height, int cpp) {
    // Get global indexes
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    int gidy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gidx >= width || gidy >= height) return;

    float r = imageIn[(gidy * width + gidx) * cpp + 0];
    float g = imageIn[(gidy * width + gidx) * cpp + 1];
    float b = imageIn[(gidy * width + gidx) * cpp + 2];

    // RGB -> YUV
    unsigned char Y = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    unsigned char U = (unsigned char)((-0.168736f * r) + (-0.331264f * g) + 0.5f * b + 128);
    unsigned char V = (unsigned char)((0.5f * r) - 0.418688f * g - 0.081312f * b + 128);

    // Apply histogram equalization using LUT
    unsigned char Y_new = d_lut[Y];

    // YUV -> RGB
    r = Y_new + 1.402f * (V - 128);
    g = Y_new - 0.344136f * (U - 128) - 0.714136f * (V - 128);
    b = Y_new + 1.772f * (U - 128);

    imageOut[(gidy * width + gidx) * cpp + 0] = min(255, max(0, (int)r));
    imageOut[(gidy * width + gidx) * cpp + 1] = min(255, max(0, (int)g));
    imageOut[(gidy * width + gidx) * cpp + 2] = min(255, max(0, (int)b));
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s input.png output.png\n", argv[0]);
        return 1;
    }

    // Load input image
    int width, height, cpp;
    unsigned char *h_imageIn = stbi_load(argv[1], &width, &height, &cpp, COLOR_CHANNELS);
    if (!h_imageIn) {
        printf("Image load failed\n"); return 1;
    }
    size_t dataSize = width * height * cpp;
    unsigned char *h_imageOut = (unsigned char*)malloc(dataSize);

    // Allocate device memory
    unsigned char *d_imageIn, *d_imageOut;
    cudaMalloc(&d_imageIn, dataSize);
    cudaMalloc(&d_imageOut, dataSize);

    dim3 blockSize_1(16, 16);
    dim3 gridSize_1((width + (blockSize_1.x-1)) / blockSize_1.x, (height + (blockSize_1.y-1)) / blockSize_1.y);
    dim3 blockSize_4(16, 16);
    dim3 gridSize_4((width + (blockSize_4.x-1)) / blockSize_4.x, (height + (blockSize_4.y-1)) / blockSize_4.y);

    // Start CUDA timing
    cudaEvent_t start, stop, t_01, t_12, t_23, t_34, t_45;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&t_01);
    cudaEventCreate(&t_12);
    cudaEventCreate(&t_23);
    cudaEventCreate(&t_34);
    cudaEventCreate(&t_45);
    cudaEventRecord(start);

    checkCudaErrors(cudaMemcpy(d_imageIn, h_imageIn, dataSize, cudaMemcpyHostToDevice));
    cudaEventRecord(t_01);
    cudaEventSynchronize(t_01);

    // Compute histogram using shared memory
    computeHistogramShared<<<gridSize_1, blockSize_1>>>(d_imageIn, width, height, cpp);
    cudaEventRecord(t_12);
    cudaEventSynchronize(t_12);
    cudaDeviceSynchronize();

    // Compute CDF using parallel scan
    blellochScan<<<1, LUMINANCE_LEVELS>>>(d_cdf, LUMINANCE_LEVELS);
    cudaEventRecord(t_23);
    cudaEventSynchronize(t_23);
    cudaDeviceSynchronize();

    // Create LUT from CDF
    computeLUT<<<1, LUMINANCE_LEVELS>>>(d_cdf, LUMINANCE_LEVELS, width * height);
    cudaEventRecord(t_34);
    cudaEventSynchronize(t_34);
    cudaDeviceSynchronize();

    // Apply equalization using LUT
    applyEqualization<<<gridSize_4, blockSize_4>>>(d_imageIn, d_imageOut, width, height, cpp);
    cudaEventRecord(t_45);
    cudaEventSynchronize(t_45);
    cudaDeviceSynchronize();

    // Copy result to host and stop timing
    checkCudaErrors(cudaMemcpy(h_imageOut, d_imageOut, dataSize, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_0;
    cudaEventElapsedTime(&ms_0, start, t_01);
    printf("Memcpy (RAM -> VRAM) time is: %0.3f milliseconds \n", ms_0);
    float ms_1;
    cudaEventElapsedTime(&ms_1, t_01, t_12);
    printf("Kernel 1 time is: %0.3f milliseconds \n", ms_1);
    float ms_2;
    cudaEventElapsedTime(&ms_2, t_12, t_23);
    printf("Kernel 2 time is: %0.3f milliseconds \n", ms_2);
    float ms_3;
    cudaEventElapsedTime(&ms_3, t_23, t_34);
    printf("Kernel 3 time is: %0.3f milliseconds \n", ms_3);
    float ms_4;
    cudaEventElapsedTime(&ms_4, t_34, t_45);
    printf("Kernel 4 time is: %0.3f milliseconds \n", ms_4);
    float ms_5;
    cudaEventElapsedTime(&ms_5, t_45, stop);
    printf("Memcpy (VRAM -> RAM) time is: %0.3f milliseconds \n", ms_5);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Total execution time is: %0.3f milliseconds \n", ms);

    // Write output image
    stbi_write_png(argv[2], width, height, cpp, h_imageOut, width * cpp);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(t_01);
    cudaEventDestroy(t_12);
    cudaEventDestroy(t_23);
    cudaEventDestroy(t_34);
    cudaEventDestroy(t_45);
    cudaFree(d_imageIn);
    cudaFree(d_imageOut);
    stbi_image_free(h_imageIn);
    free(h_imageOut);
    return 0;
}

