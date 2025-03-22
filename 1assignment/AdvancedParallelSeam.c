// greedy_triangle_seam.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <limits.h>
#include <float.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define COLOR_CHANNELS 0

// Triangle wavefront cumulative energy update
void compute_triangle_cumulative_energy(float *energy, float *cumulative, int width, int height) {
    #pragma omp parallel for
    for (int j = 0; j < width; j++) {
        cumulative[j] = energy[j];
    }

    for (int i = 1; i < height; i++) {
        #pragma omp parallel for
        for (int j = 0; j < width; j++) {
            float left = (j > 0) ? cumulative[(i - 1) * width + (j - 1)] : FLT_MAX;
            float up = cumulative[(i - 1) * width + j];
            float right = (j < width - 1) ? cumulative[(i - 1) * width + (j + 1)] : FLT_MAX;

            float min_energy = fminf(fminf(left, up), right);
            cumulative[i * width + j] = energy[i * width + j] + min_energy;
        }
    }
}

// Greedy selection of k vertical seams
void find_k_seams(float *cumulative, int width, int height, int k, int **seams) {
    int *used = (int *)calloc(width, sizeof(int));

    for (int s = 0; s < k; s++) {
        float min_val = FLT_MAX;
        int min_pos = -1;

        for (int j = 0; j < width; j++) {
            if (!used[j] && cumulative[j] < min_val) {
                min_val = cumulative[j];
                min_pos = j;
            }
        }

        seams[s][0] = min_pos;
        used[min_pos] = 1;

        for (int i = 1; i < height; i++) {
            int prev = seams[s][i - 1];
            int next_col = prev;
            float next_val = cumulative[i * width + prev];

            if (prev > 0 && cumulative[i * width + (prev - 1)] < next_val) {
                next_val = cumulative[i * width + (prev - 1)];
                next_col = prev - 1;
            }
            if (prev < width - 1 && cumulative[i * width + (prev + 1)] < next_val) {
                next_val = cumulative[i * width + (prev + 1)];
                next_col = prev + 1;
            }
            seams[s][i] = next_col;
        }
    }
    free(used);
}

// Remove k seams from the image
void remove_k_seams(unsigned char *image, unsigned char *output, int width, int height, int cpp, int **seams, int k) {
    #pragma omp parallel for
    for (int i = 0; i < height; i++) {
        int *mask = (int *)calloc(width, sizeof(int));
        for (int s = 0; s < k; s++) {
            mask[seams[s][i]] = 1;
        }
        int dst = 0;
        for (int j = 0; j < width; j++) {
            if (mask[j]) continue;
            for (int c = 0; c < cpp; c++) {
                output[(i * (width - k) + dst) * cpp + c] = image[(i * width + j) * cpp + c];
            }
            dst++;
        }
        free(mask);
    }
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("Usage: %s input.png output.png num_seams\n", argv[0]);
        return 1;
    }

    int width, height, cpp;
    unsigned char *input = stbi_load(argv[1], &width, &height, &cpp, COLOR_CHANNELS);
    if (!input) {
        printf("Image load failed!\n");
        return 1;
    }

    cpp = 3; // force RGB
    float *energy = (float *)malloc(width * height * sizeof(float));
    float *cumulative = (float *)malloc(width * height * sizeof(float));
    unsigned char *output = (unsigned char *)malloc((width - atoi(argv[3])) * height * cpp);

    // Compute energy (basic gradient)
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float e = 0.0f;
            for (int c = 0; c < cpp; c++) {
                int left = (j > 0) ? input[(i * width + j - 1) * cpp + c] : input[(i * width + j) * cpp + c];
                int right = (j < width - 1) ? input[(i * width + j + 1) * cpp + c] : input[(i * width + j) * cpp + c];
                int up = (i > 0) ? input[((i - 1) * width + j) * cpp + c] : input[(i * width + j) * cpp + c];
                int down = (i < height - 1) ? input[((i + 1) * width + j) * cpp + c] : input[(i * width + j) * cpp + c];
                e += (right - left) * (right - left) + (down - up) * (down - up);
            }
            energy[i * width + j] = sqrtf(e);
        }
    }

    compute_triangle_cumulative_energy(energy, cumulative, width, height);

    int k = atoi(argv[3]);
    int **seams = (int **)malloc(k * sizeof(int *));
    for (int s = 0; s < k; s++) seams[s] = (int *)malloc(height * sizeof(int));

    find_k_seams(cumulative, width, height, k, seams);

    remove_k_seams(input, output, width, height, cpp, seams, k);

    stbi_write_png(argv[2], width - k, height, cpp, output, (width - k) * cpp);

    stbi_image_free(input);
    free(energy);
    free(cumulative);
    for (int s = 0; s < k; s++) free(seams[s]);
    free(seams);
    free(output);

    printf("Greedy triangle-based seam removal complete.\n");
    return 0;
}