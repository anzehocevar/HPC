#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// Use 0 to retain the original number of color channels
#define COLOR_CHANNELS 0

void copy_image(unsigned char *image_out, const unsigned char *image_in, size_t size)
{

    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i)
    {
        image_out[i] = image_in[i];
    }
}

int main(int argc, char *argv[])
{

    if (argc < 3)
    {
        printf("USAGE: sample input_image output_image\n");
        exit(EXIT_FAILURE);
    }

    char image_in_name[255];
    char image_out_name[255];

    snprintf(image_in_name, 255, "%s", argv[1]);
    snprintf(image_out_name, 255, "%s", argv[2]);

    // Load image from file and allocate space for the output image
    int width, height, cpp;
    unsigned char *image_in = stbi_load(image_in_name, &width, &height, &cpp, COLOR_CHANNELS);

    if (image_in == NULL)
    {
        printf("Error reading loading image %s!\n", image_in_name);
        exit(EXIT_FAILURE);
    }
    printf("Loaded image %s of size %dx%d.\n", image_in_name, width, height);
    const size_t datasize = width * height * cpp * sizeof(unsigned char);
    unsigned char *image_out = (unsigned char *)malloc(datasize);
    
    //Print the number of threads
    #pragma omp parallel
    {
        #pragma omp single
        printf("Using %d threads",omp_get_num_threads());
    }

    // Just copy the input image into output
    double start = omp_get_wtime();
    copy_image(image_out, image_in, datasize);
    double stop = omp_get_wtime();
    printf(" -> time to copy: %f s\n",stop-start);
    // Write the output image to file
    char image_out_name_temp[255];
    strncpy(image_out_name_temp, image_out_name, 255);
    char *token = strtok(image_out_name_temp, ".");
    char *file_type = NULL;
    while (token != NULL)
    {
        file_type = token;
        token = strtok(NULL, ".");
    }

    // sbatch run.sh

    // z uporabo openmp naredi:
    // 1. racunanje energije v sliki (sobel)
    // 2. racunanje cen poti (od vrha do dna slike)
    // 3. najti pot z najmanjso energijo (odstrani en pixel v vsaki vrstici)

    // sobel: uzamemo sosednje pixle
    // number of steps
    for(int steps = 0;steps < 128;steps++){

        // zracunanje energije
        double start = omp_get_wtime();
        #pragma omp parallel for
        for(int i = 0;i < height;i++){
            #pragma omp parallel for
            for(int j = 0;j < width;j++){
                double dx = 0;
                double dy = 0;
                // ko gremo izven slike, preslikaj vrednosti na rob
                // za vsak kanal posebej in vzamej vsoto oz. povprecje za energijo

                // calculate dx
                

                // calculate dy
                

                double energy = sqrt(dx*dx + dy*dy);
            }

            // racunanje cen poti
            // za vsako vrstico v sliki
            // paraleliziramo lahko vrednost vsake vrstice, ne pa celotne slike
            // razdelimo sliko na pasovne namesto na vrstice
            // vsaka nit obdela en trikotnik
            // zracunamo pravilno obrnjene trikotnik
            // nato lahko se ostale trikotnike
            // vecji trikotnik = manj rezije (vec dela za vsako nit)
            // najdi najboljso velikost za pohitritev
            

        }
    }

    // iskanje poti ni paralelno
    // kopiranje poti lahko tudi paraleliziramo


    // obstajajo aproximacije (zmamnjsa kvaliteto): odstranimo N najmanjsih poti
    // problem pri prekrivanju poti
    // lahko ponovno razdelimo na trikotnike/pasove
    // ALI: markiramo pixle, in ce se prekrivajo vzamemo sosednjega




    if (!strcmp(file_type, "png"))
        stbi_write_png(image_out_name, width, height, cpp, image_out, width * cpp);
    else if (!strcmp(file_type, "jpg"))
        stbi_write_jpg(image_out_name, width, height, cpp, image_out, 100);
    else if (!strcmp(file_type, "bmp"))
        stbi_write_bmp(image_out_name, width, height, cpp, image_out);
    else
        printf("Error: Unknown image format %s! Only png, bmp, or bmp supported.\n", file_type);

    // Release the memory
    free(image_in);
    free(image_out);

    return 0;
}
