#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <limits.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// Use 0 to retain the original number of color channels
#define COLOR_CHANNELS 0

void copy_image(unsigned char *image_out, const unsigned char *image_in, size_t size) {
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        image_out[i] = image_in[i];
    }
}

int main(int argc, char *argv[]) {

    if (argc < 3) {
        printf("USAGE: sample input_image output_image\n");
        exit(EXIT_FAILURE);
    }

    char image_in_name[255];
    char image_out_name[255];

    snprintf(image_in_name, 255, "%s", argv[1]);
    snprintf(image_out_name, 255, "%s", argv[2]);

    int width, height, cpp;
    unsigned char *image_in = stbi_load(image_in_name, &width, &height, &cpp, COLOR_CHANNELS);

    if (image_in == NULL) {
        printf("Error loading image %s!\n", image_in_name);
        exit(EXIT_FAILURE);
    }

    printf("Loaded image %s of size %dx%d.\n", image_in_name, width, height);
    size_t datasize = (size_t)width * height * cpp;
    unsigned char *image_out = (unsigned char *)malloc(datasize * sizeof(unsigned char));
    unsigned char *energy = (unsigned char *)malloc(width * height * sizeof(unsigned char));

    if (!image_out || !energy) {
        printf("Memory allocation failed!\n");
        stbi_image_free(image_in);
        exit(EXIT_FAILURE);
    }

    #pragma omp parallel
    {
        #pragma omp single
        printf("Using %d threads\n", omp_get_num_threads());
    }

    // Copy the image
    copy_image(image_out, image_in, datasize);

    // Energy Calculation
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            double Gx[3] = {0.0, 0.0, 0.0};
            double Gy[3] = {0.0, 0.0, 0.0};

            for (int c = 0; c < cpp; c++) {
                // Handle out of bounds conditions
                int i_minus_1 = (i - 1 < 0) ? 0 : i - 1;
                int i_plus_1 = (i + 1 >= height) ? height - 1 : i + 1;
                int j_minus_1 = (j - 1 < 0) ? 0 : j - 1;
                int j_plus_1 = (j + 1 >= width) ? width - 1 : j + 1;
    
                // Sobel operator 
                Gx[c] = -image_out[i_minus_1 * width * cpp + j_minus_1 * cpp + c]
                        - 2 * image_out[i * width * cpp + j_minus_1 * cpp + c]
                        - image_out[i_plus_1 * width * cpp + j_minus_1 * cpp + c]
                        + image_out[i_minus_1 * width * cpp + j_plus_1 * cpp + c]
                        + 2 * image_out[i * width * cpp + j_plus_1 * cpp + c]
                        + image_out[i_plus_1 * width * cpp + j_plus_1 * cpp + c];
    
                Gy[c] = image_out[i_minus_1 * width * cpp + j_minus_1 * cpp + c]
                        + 2 * image_out[i_minus_1 * width * cpp + j * cpp + c]
                        + image_out[i_minus_1 * width * cpp + j_plus_1 * cpp + c]
                        - image_out[i_plus_1 * width * cpp + j_minus_1 * cpp + c]
                        - 2 * image_out[i_plus_1 * width * cpp + j * cpp + c]
                        - image_out[i_plus_1 * width * cpp + j_plus_1 * cpp + c];
            }

            double energy_val = (sqrt(Gx[0] * Gx[0] + Gy[0] * Gy[0]) +
                                 sqrt(Gx[1] * Gx[1] + Gy[1] * Gy[1]) +
                                 sqrt(Gx[2] * Gx[2] + Gy[2] * Gy[2])) / 3.0;
            
            energy[i * width + j] = (unsigned char)energy_val;
        }
    }

    // Grayscale enery image
    // Normalize energy values to 0-255 range
    unsigned char *energy_image = (unsigned char *)malloc(width * height * sizeof(unsigned char));
    double max_energy = 0.0;

    // Find the maximum energy value for normalization
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (energy[i * width + j] > max_energy) {
                max_energy = energy[i * width + j];
            }
        }
    }

    // Scale energy values to fit in 0-255
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            energy_image[i * width + j] = (unsigned char)((energy[i * width + j] / max_energy) * 255);
        }
    }

    // Save the energy image as a grayscale PNG
    stbi_write_png("energy.png", width, height, 1, energy_image, width);

    // Free allocated memory for energy image
    free(energy_image);

    printf("Energy image saved as energy.png\n");

    // Update final image size and save the result
    stbi_write_png(image_out_name, width, height, cpp, image_out, width * cpp);


    stbi_image_free(image_in);
    free(image_out);
    free(energy);

    printf("Finished seam carving.\n");
    return 0;
}
