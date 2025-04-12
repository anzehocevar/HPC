#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <sys/param.h>
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// Use 0 to retain the original number of color channels
#define COLOR_CHANNELS 0

#define LUMINANCE_LEVELS 256
int histogram[LUMINANCE_LEVELS];
int histogramCumulative[LUMINANCE_LEVELS];

void copy_image(unsigned char *dst, const unsigned char *src, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        dst[i] = src[i];
    }
}

void histogram_equalization(unsigned char* imageIn, unsigned char* imageOut, unsigned char* yuvCache, int height, int width, int cpp) {

    // Build luminance histogram
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Read RGB
            float red   = (float) imageIn[(y * width + x) * cpp + 0];
            float green = (float) imageIn[(y * width + x) * cpp + 1];
            float blue  = (float) imageIn[(y * width + x) * cpp + 2];
            // RGB -> YUV
            unsigned char Y = (unsigned char) (0.299 * red + 0.587 * green + 0.114 * blue + 0);
            unsigned char U = (unsigned char) ((-0.168736 * red) + (-0.331264 * green) + 0.5 * blue + 128);
            unsigned char V = (unsigned char) (0.5 * red + (-0.418688 * green) + (-0.081312 * blue) + 128);
            // Save YUV values for other loop
            yuvCache[(y * width + x) * cpp + 0] = Y;
            yuvCache[(y * width + x) * cpp + 1] = U;
            yuvCache[(y * width + x) * cpp + 2] = V;
            // Update histogram
            histogram[Y] += 1;
        }
    }

    // Build cumulative histogram
    for (int i = 1; i < LUMINANCE_LEVELS; i++)
        histogramCumulative[i] = histogram[i] + histogramCumulative[i-1];

    // Find minumum non-zero value in the histogram
    unsigned char minCdf = 0;
    for (int i = 0; i < LUMINANCE_LEVELS && minCdf < 1; i++)
        minCdf = histogramCumulative[i];
    float minCdf_f = (float) minCdf;

    // Calculate new luminance level & convert back to RGB
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Read YUV from cache
            unsigned char Y = yuvCache[(y * width + x) * cpp + 0];
            unsigned char U = yuvCache[(y * width + x) * cpp + 1];
            unsigned char V = yuvCache[(y * width + x) * cpp + 2];
            // New luminance level
            unsigned char Y_new = (unsigned char) ((histogramCumulative[Y] - minCdf_f)/(height*width - minCdf_f) * (LUMINANCE_LEVELS-1.0));
            // YUV -> RGB
            imageOut[(y * width + x) * cpp + 0] = MIN(
                LUMINANCE_LEVELS-1,
                MAX(
                    0,
                    (Y_new + 1.402 * (V-128))
                )
            );
            imageOut[(y * width + x) * cpp + 1] = MIN(
                LUMINANCE_LEVELS-1,
                MAX(
                    0,
                    (Y_new - 0.344136 * (U-128) - 0.714136 * (V-128))
                )
            );
            imageOut[(y * width + x) * cpp + 2] = MIN(
                LUMINANCE_LEVELS-1,
                MAX(
                    0,
                    (Y_new + 1.772 * (U-128))
                )
            );
        }
    }

}

int main(int argc, char *argv[]) {

    if (argc < 3) {
        printf("USAGE: %s input_image output_image\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    // Read filenames from command line arguments
    char imageInName[255];
    char imageOutName[255];
    snprintf(imageInName, 255, "%s", argv[1]);
    snprintf(imageOutName, 255, "%s", argv[2]);
    
    // Read image
    int width, height, cpp;
    unsigned char *imageIn = stbi_load(imageInName, &width, &height, &cpp, COLOR_CHANNELS);

    if (imageIn == NULL) {
        printf("Error loading image %s!\n", imageInName);
        exit(EXIT_FAILURE);
    }

    printf("Loaded image %s of size %dx%d.\n", imageInName, width, height);
    size_t datasize = (size_t)width * height * cpp;
    unsigned char *imageOut = (unsigned char *)malloc(datasize * sizeof(unsigned char));

    if (!imageOut) {
        printf("Memory allocation failed!\n");
        stbi_image_free(imageIn);
        exit(EXIT_FAILURE);
    }

    // Run histogram equalization
    unsigned char* yuvCache = (unsigned char*) malloc(sizeof(unsigned char) * height * width * 3);
    double t0 = omp_get_wtime();
    histogram_equalization(imageIn, imageOut, yuvCache, height, width, cpp);
    double t_total_ms = 1000*(omp_get_wtime() - t0);
    printf("Total execution time is: %0.3f milliseconds \n", t_total_ms);

    // Update final image size and save the result
    stbi_write_png(imageOutName, width, height, cpp, imageOut, width * cpp);
    printf("Saved image as %s\n", imageOutName);

    // Free allocated variables
    stbi_image_free(imageIn);
    free(yuvCache);
    free(imageOut);

    return 0;
}
