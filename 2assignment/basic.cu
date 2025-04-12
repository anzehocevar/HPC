#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include "helper_cuda.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define COLOR_CHANNELS 0
#define DEBUG_MODE 0

#define BLOCK_SIZE_X_1 8
#define BLOCK_SIZE_Y_1 8
// #define BLOCK_SIZE_X_2 32
// #define BLOCK_SIZE_Y_2 32
#define BLOCK_SIZE_X_3 8
#define BLOCK_SIZE_Y_3 8

#define LUMINANCE_LEVELS 256
__device__ int d_histogram[LUMINANCE_LEVELS];
__device__ int d_histogramCumulative[LUMINANCE_LEVELS];

__global__ void dummyKernel() {
    // empty warm-up kernel
}

__global__ void computeHistogram(const unsigned char *imageIn, const int width, const int height, const int cpp)
{

    // Get global indexes
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;
    if (DEBUG_MODE && gidx == 0 && gidy == 0) {
        printf("DEVICE: Computing histogram\n");
    }

    if (gidx < width && gidy < height) {
        // Read RGB
        float red   = (float) imageIn[(gidy * width + gidx) * cpp + 0];
        float green = (float) imageIn[(gidy * width + gidx) * cpp + 1];
        float blue  = (float) imageIn[(gidy * width + gidx) * cpp + 2];

        // RGB -> YUV
        unsigned char Y = (unsigned char) (0.299 * red + 0.587 * green + 0.114 * blue + 0);
        // unsigned char U = (unsigned char) ((-0.168736 * red) + (-0.331264 * green) + 0.5 * blue + 128);
        // unsigned char V = (unsigned char) (0.5 * red + (-0.418688 * green) + (-0.081312 * blue) + 128);

        atomicAdd(&(d_histogram[Y]), 1);
    }

}

__global__ void computeHistogramCumulative(const int width, const int height, const int cpp) {
    // Get global indexes
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;
    if (gidx >= width || gidy >= height)
        return;
    if (DEBUG_MODE && gidx == 0 && gidy == 0) {
        printf("DEVICE: Computing cumulative histogram\n");
    }
    if (gidx == 0 & gidy == 0) {
        d_histogramCumulative[0] = d_histogram[0];
        for (int i = 1; i < LUMINANCE_LEVELS; i++)
            d_histogramCumulative[i] = d_histogram[i] + d_histogramCumulative[i-1];
        // for (int i = 0; i < LUMINANCE_LEVELS; i++)
        //     printf("%d, ", d_histogramCumulative[i]);
    }
}

__global__ void computeNewLuminance(const unsigned char *imageIn, unsigned char *imageOut, const int width, const int height, const int cpp) {

    // Get global indexes
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;
    if (gidx >= width || gidy >= height)
        return;
    if (DEBUG_MODE && gidx == 0 && gidy == 0) {
        printf("DEVICE: Computing new luminance levels\n");
    }

    // Read RGB
    float red   = (float) imageIn[(gidy * width + gidx) * cpp + 0];
    float green = (float) imageIn[(gidy * width + gidx) * cpp + 1];
    float blue  = (float) imageIn[(gidy * width + gidx) * cpp + 2];

    // RGB -> YUV
    unsigned char Y = (unsigned char) (0.299 * red + 0.587 * green + 0.114 * blue + 0);
    unsigned char U = (unsigned char) ((-0.168736 * red) + (-0.331264 * green) + 0.5 * blue + 128);
    unsigned char V = (unsigned char) (0.5 * red + (-0.418688 * green) + (-0.081312 * blue) + 128);

    // Find minumum non-zero value in the histogram
    unsigned char minCdf = 0;
    for (int i = 0; i < LUMINANCE_LEVELS && minCdf < 1; i++)
        minCdf = d_histogramCumulative[i];
    float minCdf_f = (float) minCdf;

    // Calculate new luminance level
    unsigned char Y_new = (unsigned char) ((d_histogramCumulative[Y] - minCdf_f)/(height*width - minCdf_f) * (LUMINANCE_LEVELS-1.0));

    // YUV -> RGB
    red = (Y_new + 1.402 * (V-128));
    green = (Y_new - 0.344136 * (U-128) - 0.714136 * (V-128));
    blue = (Y_new + 1.772 * (U-128));
    imageOut[(gidy * width + gidx) * cpp + 0] = MIN(LUMINANCE_LEVELS-1, MAX(0, red));
    imageOut[(gidy * width + gidx) * cpp + 1] = MIN(LUMINANCE_LEVELS-1, MAX(0, green));
    imageOut[(gidy * width + gidx) * cpp + 2] = MIN(LUMINANCE_LEVELS-1, MAX(0, blue));

}

int main(int argc, char *argv[])
{

    if (argc < 3)
    {
        printf("USAGE: %s input_image output_image\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    char szImage_in_name[255];
    char szImage_out_name[255];

    snprintf(szImage_in_name, 255, "%s", argv[1]);
    snprintf(szImage_out_name, 255, "%s", argv[2]);

    // Load image from file and allocate space for the output image
    int width, height, cpp;
    unsigned char *h_imageIn = stbi_load(szImage_in_name, &width, &height, &cpp, COLOR_CHANNELS);

    if (h_imageIn == NULL)
    {
        printf("Error reading loading image %s!\n", szImage_in_name);
        exit(EXIT_FAILURE);
    }
    printf("Loaded image %s of size %dx%d.\n", szImage_in_name, width, height);
    const size_t datasize = width * height * cpp * sizeof(unsigned char);
    unsigned char *h_imageOut = (unsigned char *)malloc(datasize);

    // Setup Thread organization
    dim3 blockSize_1(BLOCK_SIZE_X_1, BLOCK_SIZE_Y_1);
    // dim3 gridSize((height-1)/blockSize_1.x+1,(width-1)/blockSize_1.y+1);
    dim3 gridSize_1((width-1)/blockSize_1.x+1,(height-1)/blockSize_1.y+1);
    //dim3 gridSize(1, 1);
    // dim3 blockSize_2(BLOCK_SIZE_X_2, BLOCK_SIZE_Y_2);
    // dim3 gridSize_2((width-1)/blockSize_2.x+1,(height-1)/blockSize_2.y+1);
    dim3 blockSize_3(BLOCK_SIZE_X_3, BLOCK_SIZE_Y_3);
    dim3 gridSize_3((width-1)/blockSize_3.x+1,(height-1)/blockSize_3.y+1);

    unsigned char *d_imageIn;
    unsigned char *d_imageOut;

    // Allocate memory on the device
    checkCudaErrors(cudaMalloc(&d_imageIn, datasize));
    checkCudaErrors(cudaMalloc(&d_imageOut, datasize));

    // CUDA MALLOC,..

    // Use CUDA events to measure execution time
    cudaEvent_t start, stop, t_01, t_12, t_23, t_34;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&t_01);
    cudaEventCreate(&t_12);
    cudaEventCreate(&t_23);
    cudaEventCreate(&t_34);

    // Warm-up GPU
    dummyKernel<<<1, 1>>>();
    // cudaDeviceSynchronize();

    // Copy image to device and run kernel
    cudaEventRecord(start);
    checkCudaErrors(cudaMemcpy(d_imageIn, h_imageIn, datasize, cudaMemcpyHostToDevice));
    cudaEventRecord(t_01);
    cudaEventSynchronize(t_01);
    computeHistogram<<<gridSize_1, blockSize_1>>>(d_imageIn, width, height, cpp);
    cudaEventRecord(t_12);
    cudaEventSynchronize(t_12);
    computeHistogramCumulative<<<1, 1>>>(width, height, cpp);
    cudaEventRecord(t_23);
    cudaEventSynchronize(t_23);
    computeNewLuminance<<<gridSize_3, blockSize_3>>>(d_imageIn, d_imageOut, width, height, cpp);
    cudaEventRecord(t_34);
    cudaEventSynchronize(t_34);
    checkCudaErrors(cudaMemcpy(h_imageOut, d_imageOut, datasize, cudaMemcpyDeviceToHost));
    // cudaDeviceSynchronize();
    getLastCudaError("Kernel execution failed\n");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Print time
    float milliseconds_0 = 0;
    cudaEventElapsedTime(&milliseconds_0, start, t_01);
    printf("Memcpy (RAM -> VRAM) time is: %0.3f milliseconds \n", milliseconds_0);
    float milliseconds_1 = 0;
    cudaEventElapsedTime(&milliseconds_1, t_01, t_12);
    printf("Kernel 1 time is: %0.3f milliseconds \n", milliseconds_1);
    float milliseconds_2 = 0;
    cudaEventElapsedTime(&milliseconds_2, t_12, t_23);
    printf("Kernel 2 time is: %0.3f milliseconds \n", milliseconds_2);
    float milliseconds_3 = 0;
    cudaEventElapsedTime(&milliseconds_3, t_23, t_34);
    printf("Kernel 3 time is: %0.3f milliseconds \n", milliseconds_3);
    float milliseconds_4 = 0;
    cudaEventElapsedTime(&milliseconds_4, t_34, stop);
    printf("Memcpy (VRAM -> RAM) time is: %0.3f milliseconds \n", milliseconds_4);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Total execution time is: %0.3f milliseconds \n", milliseconds);

    // Write the output file
    char szImage_out_name_temp[255];
    strncpy(szImage_out_name_temp, szImage_out_name, 255);
    char *token = strtok(szImage_out_name_temp, ".");
    char *FileType = NULL;
    while (token != NULL)
    {
        FileType = token;
        token = strtok(NULL, ".");
    }

    if (!strcmp(FileType, "png"))
        stbi_write_png(szImage_out_name, width, height, cpp, h_imageOut, width * cpp);
    else if (!strcmp(FileType, "jpg"))
        stbi_write_jpg(szImage_out_name, width, height, cpp, h_imageOut, 100);
    else if (!strcmp(FileType, "bmp"))
        stbi_write_bmp(szImage_out_name, width, height, cpp, h_imageOut);
    else
        printf("Error: Unknown image format %s! Only png, bmp, or bmp supported.\n", FileType);

    // Free device memory
    checkCudaErrors(cudaFree(d_imageIn));
    checkCudaErrors(cudaFree(d_imageOut));

    // Clean-up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(t_01);
    cudaEventDestroy(t_12);
    cudaEventDestroy(t_23);
    cudaEventDestroy(t_34);

    // Free host memory
    free(h_imageIn);
    free(h_imageOut);

    return 0;
}
