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

    
    // Copy the image
    copy_image(image_out, image_in, datasize);

    for(int steps = 0;steps < 128;steps++){
        // calculate energy
        for(int i = 0;i < height;i++){
            for(int j = 0;j < width;j++){
                double Gx = -image_out[(i - 1) * width * cpp + (j - 1) * cpp]
                        - 2 * image_out[i * width * cpp + (j - 1) * cpp]
                        - image_out[(i + 1) * width * cpp + (j - 1) * cpp]
                        + image_out[(i - 1) * width * cpp + (j + 1) * cpp]
                        + 2 * image_out[i * width * cpp + (j + 1) * cpp]
                        + image_out[(i + 1) * width * cpp + (j + 1) * cpp];

                double Gy = image_out[(i - 1) * width * cpp + (j - 1) * cpp] 
                        + 2 * image_out[(i - 1) * width * cpp + j * cpp] 
                        + image_out[(i - 1) * width * cpp + (j + 1) * cpp]
                        - image_out[(i + 1) * width * cpp + (j - 1) * cpp] 
                        - 2 * image_out[(i + 1) * width * cpp + j * cpp] 
                        - image_out[(i + 1) * width * cpp + (j + 1) * cpp];
    
                double energy = sqrt(Gx * Gx + Gy * Gy);
                image_out[i * width * cpp + j * cpp] = (unsigned char)energy;
            }
        }

        // remember path
        int *path = (int *)malloc(sizeof(int) * height);

        // calculate path cost
        int cost = INT_MAX;
        int col = 0;
        for (int i = 0; i < width; i++)
        {
            if (image_out[(height - 1) * width * cpp + i * cpp] < cost)
            {
                cost = image_out[(height - 1) * width * cpp + i * cpp];
                col = i;
            }
        }


        path[0] = col;
        for (int i = 1; i < height; i++)
        {
            int min = INT_MAX;
            int min_col = 0;
            // look at neighbours
            for (int j = -1; j < 2; j++)
            {
                if (col + j >= 0 && col + j < width)
                {
                    if (image_out[(height - i) * width * cpp + (col + j) * cpp] < min)
                    {
                        min = image_out[(height - i) * width * cpp + (col + j) * cpp];
                        min_col = col + j;
                    }
                }
            }
            col = min_col;
            path[i] = col;
        }

        // remove the cheapest path
        // but what happens when the talk isnt cheap
        // and it is what it is?
        // what i mean by that?
        // hillary clinton is a lizard
        //

        // remove the cheapest path
        for (int i = 0; i < height; i++)
        {
            for (int j = path[i]; j < width - 1; j++)
            {
                for (int c = 0; c < cpp; c++)
                {
                    image_out[i * width * cpp + j * cpp + c] = image_out[i * width * cpp + (j + 1) * cpp + c];
                }
            }
        }
        width--;

        // remove the path
        free(path);
        

    }
    




    if (!strcmp(file_type, "png"))
        stbi_write_png(image_out_name, width, height, cpp, image_out, width * cpp);
    else if (!strcmp(file_type, "jpg"))
        stbi_write_jpg(image_out_name, width, height, cpp, image_out, 100);
    else if (!strcmp(file_type, "bmp"))
        stbi_write_bmp(image_out_name, width, height, cpp, image_out);
    else
        printf("Error: Unknown image format %s! Only png, bmp, or bmp supported.\n", file_type);


    // Free the image
    stbi_image_free(image_in);
    free(image_in);
    free(image_out);

    return 0;
}


