#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <limits.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define DEBUG_MODE 1
#define NUM_OF_SEAMS_DEFAULT 128
#include "stb_image.h"
#include "stb_image_write.h"

// Use 0 to retain the original number of color channels
#define COLOR_CHANNELS 0

void copy_image(unsigned char *dst, const unsigned char *src, size_t size) {
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        dst[i] = src[i];
    }
}

void find_vertical_seam(int* seam, unsigned char* energy, int width, int height) {
    // Find lowest value in top row
    int min_position = 0;
    int min_energy = INT_MAX;
    for(int i = 0;i < width;i++){
        if(energy[i] < min_energy){
            min_energy = energy[i];
            min_position = i;
        }
    }
    seam[0] = min_position;

    // Iteratively select the lowest energy path
    for(int i = 1;i < height;i++){
        int next_position = min_position;
        int next_energy = INT_MAX;
        for(int j = -1;j < 2;j++){
            // bounds check
            if(min_position + j >= 0 && min_position + j < width){
                if(energy[(i + 1) * width + min_position + j] < next_energy){
                    next_energy = energy[(i + 1) * width + min_position + j];
                    next_position = min_position + j;
                }
            }
        }
        seam[i] = next_position;
        min_position = next_position;
    }
}

// Grayscale enery image
// Normalize energy values to 0-255 range
unsigned char* calc_energy_image(unsigned char* energy_image, unsigned char* energy, int width, int height) {
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

    return energy_image;
}

void calc_energy(unsigned char* input, unsigned char* energy, int width, int height, int cpp) {
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
                Gx[c] = -input[i_minus_1 * width * cpp + j_minus_1 * cpp + c]
                        - 2 * input[i * width * cpp + j_minus_1 * cpp + c]
                        - input[i_plus_1 * width * cpp + j_minus_1 * cpp + c]
                        + input[i_minus_1 * width * cpp + j_plus_1 * cpp + c]
                        + 2 * input[i * width * cpp + j_plus_1 * cpp + c]
                        + input[i_plus_1 * width * cpp + j_plus_1 * cpp + c];
    
                Gy[c] = input[i_minus_1 * width * cpp + j_minus_1 * cpp + c]
                        + 2 * input[i_minus_1 * width * cpp + j * cpp + c]
                        + input[i_minus_1 * width * cpp + j_plus_1 * cpp + c]
                        - input[i_plus_1 * width * cpp + j_minus_1 * cpp + c]
                        - 2 * input[i_plus_1 * width * cpp + j * cpp + c]
                        - input[i_plus_1 * width * cpp + j_plus_1 * cpp + c];
            }

            double energy_val = (sqrt(Gx[0] * Gx[0] + Gy[0] * Gy[0]) +
                                 sqrt(Gx[1] * Gx[1] + Gy[1] * Gy[1]) +
                                 sqrt(Gx[2] * Gx[2] + Gy[2] * Gy[2])) / 3.0;
            
            energy[i * width + j] = (unsigned char)energy_val;
        }
    }

}

void calc_image_narrower(unsigned char* image_narrower, unsigned char* image, int* seam, int width, int height, int cpp) {
    int width_minus_1 = width - 1;  // To make it abundantly clear
    for (int i = 0; i < height; i++) {
        int seam_col = seam[i];
        int dst_col = 0;
        for (int j = 0; j < width; j++) {  // original width before decrement
            if (j == seam_col) continue;  // skip the seam pixel

            for (int c = 0; c < cpp; c++) {
                image_narrower[(i * width_minus_1 + dst_col) * cpp + c] =
                    image[(i * width + j) * cpp + c];
            }
            dst_col++;
        }
    }
}

int main(int argc, char *argv[]) {

    if (argc < 3) {
        printf("USAGE: sample input_image output_image\n");
        exit(EXIT_FAILURE);
    }

    // Read filenames from command line arguments
    char image_in_name[255];
    char image_out_name[255];
    snprintf(image_in_name, 255, "%s", argv[1]);
    snprintf(image_out_name, 255, "%s", argv[2]);
    
    // Determine number of seams to remove
    int num_of_seams = NUM_OF_SEAMS_DEFAULT;
    if (argc > 3)
        num_of_seams = (int) strtol(argv[3], NULL, 10);
    printf("Number of seams: %d\n", num_of_seams);

    // Determine number of threads to run
    int num_of_threads = 0;
    if (argc > 4) {
        num_of_threads = (int) strtol(argv[4], NULL, 10);
        omp_set_num_threads(num_of_threads);
    }

    // Read image
    int width, height, cpp;
    unsigned char *image_in = stbi_load(image_in_name, &width, &height, &cpp, COLOR_CHANNELS);

    if (image_in == NULL) {
        printf("Error loading image %s!\n", image_in_name);
        exit(EXIT_FAILURE);
    }

    printf("Loaded image %s of size %dx%d.\n", image_in_name, width, height);
    size_t datasize = (size_t)width * height * cpp;
    unsigned char *image = (unsigned char *)malloc(datasize * sizeof(unsigned char));

    if (!image) {
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
    copy_image(image, image_in, datasize);

    // Save location of pixels with lowest path
    int* seam = (int*) malloc(height * sizeof(int));

    // Variables for total time
    double t_start = omp_get_wtime();
    double t_energy = 0.0;
    double t_identification = 0.0;
    double t_removal = 0.0;
    double t_copy = 0.0;

    // seam - path from top to bottom with lowest Energy
    // solve with dynamic programming
    for(int iter = 0;iter < num_of_seams;iter++){

        if (DEBUG_MODE)
            printf("Iteration [%d/%d]\n", iter+1, num_of_seams);

        unsigned char *energy = (unsigned char *)malloc(width * height * sizeof(unsigned char));

        if (!energy) {
            printf("Memory allocation failed!\n");
            stbi_image_free(image_in);
            exit(EXIT_FAILURE);
        }

        // measure time
        double t_energy_start = omp_get_wtime();

        // Energy Calculation
        calc_energy(image, energy, width, height, cpp);

        double t_energy_end = omp_get_wtime();
        t_energy += t_energy_end - t_energy_start;

        if (DEBUG_MODE && iter == 0) {
            // Grayscale energy image
            // Normalize energy values to 0-255 range
            unsigned char *energy_image = (unsigned char *)malloc(width * height * sizeof(unsigned char));
            calc_energy_image(energy_image, energy, width, height);

            // Save the energy image as a grayscale PNG
            stbi_write_png("energy.png", width, height, 1, energy_image, width);

            // Free allocated memory for energy image
            free(energy_image);

            printf("Energy image saved as energy.png\n");
        }

        double t_identification_start = omp_get_wtime();

        // Vertical seam identification
        // start at the bottom
        for (int i = height - 2; i >= 0; i--) {
            for (int j = 0; j < width; j++) {
                int j_minus_1 = (j - 1 < 0) ? j : j - 1;
                int j_plus_1 = (j + 1 >= width) ? j : j + 1;
        
                int below_left = energy[(i + 1) * width + j_minus_1];
                int below = energy[(i + 1) * width + j];
                int below_right = energy[(i + 1) * width + j_plus_1];
        
                energy[i * width + j] += fmin(fmin(below_left, below), below_right);
            }
        }

        double t_identification_end = omp_get_wtime();
        t_identification += t_identification_end - t_identification_start;

        double t_removal_start = omp_get_wtime();

        // Vertical seam removal
        // Fill array seam with column indexes of path
        find_vertical_seam(seam, energy, width, height);

        double t_removal_end = omp_get_wtime();
        t_removal += t_removal_end - t_removal_start;

        double t_copy_start = omp_get_wtime();

        // Create new image by skipping seam pixels in current image
        unsigned char *image_narrower = (unsigned char *)malloc((width-1) * height * cpp);
        calc_image_narrower(image_narrower, image, seam, width, height, cpp);

        double t_copy_end = omp_get_wtime();
        t_copy += t_copy_end - t_copy_start;

        // Replace old image
        free(image);
        free(energy);
        image = image_narrower;
        // Update image width after seam removal
        width -= 1;
    }

    printf("Finished seam carving.\n");

    // Update final image size and save the result
    stbi_write_png(image_out_name, width, height, cpp, image, width * cpp);
    printf("Saved image as %s\n", image_out_name);

    // Free allocated variables
    stbi_image_free(image_in);
    free(image);
    free(seam);

    double t_end = omp_get_wtime();
    printf("Energy calculation took %.3f seconds\n", t_energy);
    printf("Vertical seam identification took %.3f seconds\n", t_identification);
    printf("Seam removal took %.3f seconds\n", t_removal);
    printf("Copying took %.3f seconds\n", t_copy);
    printf("Total time: %.3f seconds\n", t_end - t_start);
    return 0;
}
