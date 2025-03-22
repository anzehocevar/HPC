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

#define COLOR_CHANNELS 0

void copy_image(unsigned char *image_out, const unsigned char *image_in, size_t size) {
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        image_out[i] = image_in[i];
    }
}

int main(int argc, char *argv[]) {

    if (argc < 3) {
        printf("USAGE: ./SequentialSeam input_image output_image\n");
        exit(EXIT_FAILURE);
    }

    char image_in_name[255], image_out_name[255];
    snprintf(image_in_name, 255, "%s", argv[1]);
    snprintf(image_out_name, 255, "%s", argv[2]);

    int width, height, cpp;
    unsigned char *image_in = stbi_load(image_in_name, &width, &height, &cpp, COLOR_CHANNELS);
    if (!image_in) {
        printf("Error loading image %s!\n", image_in_name);
        exit(EXIT_FAILURE);
    }

    printf("Loaded image %s of size %dx%d.\n", image_in_name, width, height);
    size_t datasize = (size_t)width * height * cpp;
    unsigned char *image_out = (unsigned char *)malloc(datasize);
    unsigned char *energy = (unsigned char *)malloc(width * height);
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

    copy_image(image_out, image_in, datasize);
    double start = omp_get_wtime();

    // ENERGY CALCULATION
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            double Gx[3] = {0.0}, Gy[3] = {0.0};

            for (int c = 0; c < cpp; c++) {
                int i1 = (i - 1 < 0) ? 0 : i - 1;
                int i2 = (i + 1 >= height) ? height - 1 : i + 1;
                int j1 = (j - 1 < 0) ? 0 : j - 1;
                int j2 = (j + 1 >= width) ? width - 1 : j + 1;

                Gx[c] = -image_out[i1 * width * cpp + j1 * cpp + c]
                        - 2 * image_out[i * width * cpp + j1 * cpp + c]
                        - image_out[i2 * width * cpp + j1 * cpp + c]
                        + image_out[i1 * width * cpp + j2 * cpp + c]
                        + 2 * image_out[i * width * cpp + j2 * cpp + c]
                        + image_out[i2 * width * cpp + j2 * cpp + c];

                Gy[c] = image_out[i1 * width * cpp + j1 * cpp + c]
                        + 2 * image_out[i1 * width * cpp + j * cpp + c]
                        + image_out[i1 * width * cpp + j2 * cpp + c]
                        - image_out[i2 * width * cpp + j1 * cpp + c]
                        - 2 * image_out[i2 * width * cpp + j * cpp + c]
                        - image_out[i2 * width * cpp + j2 * cpp + c];
            }

            double energy_val = (sqrt(Gx[0]*Gx[0] + Gy[0]*Gy[0]) +
                                 sqrt(Gx[1]*Gx[1] + Gy[1]*Gy[1]) +
                                 sqrt(Gx[2]*Gx[2] + Gy[2]*Gy[2])) / 3.0;

            energy[i * width + j] = (unsigned char)energy_val;
        }
    }

    double energy_end = omp_get_wtime();
    printf("Energy calculation took %f seconds\n", energy_end - start);

    // ENERGY NORMALIZATION
    unsigned char *energy_image = (unsigned char *)malloc(width * height);
    double max_energy = 0.0;

    #pragma omp parallel for reduction(max:max_energy)
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (energy[i * width + j] > max_energy)
                max_energy = energy[i * width + j];
        }
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            energy_image[i * width + j] = (unsigned char)((energy[i * width + j] / max_energy) * 255);
        }
    }

    stbi_write_png("energy.png", width, height, 1, energy_image, width);
    free(energy_image);
    printf("Energy image saved as energy.png\n");

    int seam[height];
    double seam_start = omp_get_wtime();

    for (int seam_count = 0; seam_count < 128; seam_count++) {
        unsigned char *energy_copy = (unsigned char *)malloc(width * height);
        memcpy(energy_copy, energy, width * height);

        // Dynamic programming update
        for (int i = height - 2; i >= 0; i--) {
            #pragma omp parallel for
            for (int j = 0; j < width; j++) {
                int j1 = (j - 1 < 0) ? j : j - 1;
                int j2 = (j + 1 >= width) ? j : j + 1;

                int down_left = energy_copy[(i+1) * width + j1];
                int down = energy_copy[(i+1) * width + j];
                int down_right = energy_copy[(i+1) * width + j2];

                energy_copy[i * width + j] += fmin(fmin(down_left, down), down_right);
            }
        }

        memcpy(energy, energy_copy, width * height);
        free(energy_copy);

        // Trace minimum seam path
        int min_idx = 0;
        int min_val = INT_MAX;
        for (int i = 0; i < width; i++) {
            if (energy[i] < min_val) {
                min_val = energy[i];
                min_idx = i;
            }
        }
        seam[0] = min_idx;

        for (int i = 1; i < height; i++) {
            int best_col = min_idx;
            int best_val = INT_MAX;

            for (int j = -1; j <= 1; j++) {
                int col = min_idx + j;
                if (col >= 0 && col < width) {
                    int val = energy[i * width + col];
                    if (val < best_val) {
                        best_val = val;
                        best_col = col;
                    }
                }
            }

            seam[i] = best_col;
            min_idx = best_col;
        }

        // Remove seam from image
        for (int i = 0; i < height; i++) {
            int s = seam[i];
            for (int j = s; j < width - 1; j++) {
                for (int c = 0; c < cpp; c++) {
                    image_out[(i * width + j) * cpp + c] = image_out[(i * width + j + 1) * cpp + c];
                }
            }
        }

        width -= 1;
        unsigned char *new_image = (unsigned char *)malloc(width * height * cpp);

        for (int i = 0; i < height; i++) {
            int s = seam[i], col_out = 0;
            for (int j = 0; j < width + 1; j++) {
                if (j == s) continue;
                for (int c = 0; c < cpp; c++) {
                    new_image[(i * width + col_out) * cpp + c] =
                        image_out[(i * (width + 1) + j) * cpp + c];
                }
                col_out++;
            }
        }

        free(image_out);
        image_out = new_image;
    }

    double seam_end = omp_get_wtime();
    printf("Seam carving took %f seconds\n", seam_end - seam_start);
    stbi_write_png(image_out_name, width, height, cpp, image_out, width * cpp);

    printf("Saved image as %s\n", image_out_name);
    stbi_image_free(image_in);
    free(image_out);
    free(energy);

    printf("Finished seam carving.\n");
    printf("Total time: %f seconds\n", seam_end - start);
    return 0;
}
