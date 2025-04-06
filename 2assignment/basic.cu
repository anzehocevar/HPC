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

#define LUMINANCE_LEVELS 256
__device__ int d_histogram[LUMINANCE_LEVELS];
__device__ int d_histogramCumulative[LUMINANCE_LEVELS];

__global__ void computeHistogram(const unsigned char *imageIn, unsigned char *imageOut, const int width, const int height, const int cpp)
{

    // Get global indexes
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;
    if (gidx >= width || gidy >= height)
        return;
    if (gidx == 0 & gidy == 0) {
        printf("DEVICE: START COPY\n");
    }

    // Read RGB
    float red   = (float) imageIn[(gidy * width + gidx) * cpp + 0];
    float green = (float) imageIn[(gidy * width + gidx) * cpp + 1];
    float blue  = (float) imageIn[(gidy * width + gidx) * cpp + 2];

    // RGB -> YUV
    unsigned char Y = (unsigned char) (0.299 * red + 0.587 * green + 0.114 * blue + 0);
    // unsigned char U = (unsigned char) ((-0.168736 * red) + (-0.331264 * green) + 0.5 * blue + 128);
    // unsigned char V = (unsigned char) (0.5 * red + (-0.418688 * green) + (-0.081312 * blue) + 128);

    atomicAdd(&(d_histogram[Y]), 1);

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
    if (gidx == 0 & gidy == 0) {
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
        printf("USAGE: sample input_image output_image\n");
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
    dim3 blockSize(16, 16);
    // dim3 gridSize((height-1)/blockSize.x+1,(width-1)/blockSize.y+1);
    dim3 gridSize((width-1)/blockSize.x+1,(height-1)/blockSize.y+1);
    //dim3 gridSize(1, 1);

    unsigned char *d_imageIn;
    unsigned char *d_imageOut;

    // Allocate memory on the device
    checkCudaErrors(cudaMalloc(&d_imageIn, datasize));
    checkCudaErrors(cudaMalloc(&d_imageOut, datasize));

    // CUDA MALLOC,..

    // Use CUDA events to measure execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Copy image to device and run kernel
    cudaEventRecord(start);
    checkCudaErrors(cudaMemcpy(d_imageIn, h_imageIn, datasize, cudaMemcpyHostToDevice));
    computeHistogram<<<gridSize, blockSize>>>(d_imageIn, d_imageOut, width, height, cpp);
    computeNewLuminance<<<gridSize, blockSize>>>(d_imageIn, d_imageOut, width, height, cpp);
    checkCudaErrors(cudaMemcpy(h_imageOut, d_imageOut, datasize, cudaMemcpyDeviceToHost));
    // cudaDeviceSynchronize();
    getLastCudaError("Kernel execution failed\n");
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    // Print time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel Execution time is: %0.3f milliseconds \n", milliseconds);
    
    // for (int i = 0; i < LUMINANCE_LEVELS; i++)
    //     printf("%d, ", h_histogramCumulative[i]);

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

    // Free host memory
    free(h_imageIn);
    free(h_imageOut);

    return 0;
}
