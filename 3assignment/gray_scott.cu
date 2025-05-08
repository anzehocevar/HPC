#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include "gray_scott.h"

// Helper macro to access 2D grid
#define IDX(i, j, size) ((i) * (size) + (j))

// Reference function for initialization of U and V


void initUV2D(float *U, float *V, int size) {
    // Set initial values: U=1.0, V=0.0
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            U[IDX(i, j, size)] = 1.0f;
            V[IDX(i, j, size)] = 0.0f;
        }
    }

    // Seed a small square in the center
    int r = size / 8;
    for (int i = size / 2 - r; i < size / 2 + r; i++) {
        for (int j = size / 2 - r; j < size / 2 + r; j++) {
            U[IDX(i, j, size)] = 0.50f;
            V[IDX(i, j, size)] = 0.25f;
        }
    }
}


double gray_scott2D(gs_config config){
    // Initialize vars from .h
    int grid_size = config.n;
    int iterations = config.steps;
    float dt = config.dt;
    float du = config.du;
    float dv = config.dv;
    float f = config.f;
    float k = config.k;

    // Allocate memory
    float *U = (float *)malloc(size * size * sizeof(float));
    float *V = (float *)malloc(size * size * sizeof(float));
    float *U_next = (float *)malloc(size * size * sizeof(float));
    float *V_next = (float *)malloc(size * size * sizeof(float));

    
    // Initialize U and V
    initUV2D(U, V, size);

    /*
    YOUR SOLUTION GOES HERE
    Write a 2D Gray-Scott simulation in C/C++ using CUDA, OpenMPI, and OpenMP.
    */

    for(int it = 0;it < iterations; it++){
        // Update U and V using the Gray-Scott model
        #pragma omp parallel for collapse(2)
        for(int i = 0;i < grid_size; i++){
            for(int j = 0;j < grid_size;j++){
                // Get the indices of the neighbors
                int up = (i - 1 + grid_size) % grid_size;
                int down = (i + 1) % grid_size;
                int left = (j - 1 + grid_size) % grid_size;
                int right = (j + 1) % grid_size;

                // Compute the Laplacian
                float laplacian_U = U[IDX(up, j, grid_size)] + U[IDX(down, j, grid_size)] +
                                    U[IDX(i, left, grid_size)] + U[IDX(i, right, grid_size)] -
                                    4 * U[IDX(i, j, grid_size)];

                float laplacian_V = V[IDX(up, j, grid_size)] + V[IDX(down, j, grid_size)] +
                                    V[IDX(i, left, grid_size)] + V[IDX(i, right, grid_size)] -
                                    4 * V[IDX(i, j, grid_size)];

                // Update U and V
                U_next[IDX(i, j, grid_size)] = U[IDX(i, j, grid_size)] +
                                                dt * (du * laplacian_U - U[IDX(i, j, grid_size)] * V[IDX(i, j, grid_size)] * V[IDX(i, j, grid_size)] +
                                                f * (1 - U[IDX(i, j, grid_size)]));

                V_next[IDX(i, j, grid_size)] = V[IDX(i, j, grid_size)] +
                                                dt * (dv * laplacian_V + U[IDX(i,j,size)] * V[IDX(i,j,size)] * V[IDX(i,j,size)] -
                                                (f + k) * V[IDX(i,j,size)]);
            }
        }

        // Swap pointers
        float *temp = U;
        U = U_next;
        U_next = temp;

        temp = V;
        V = V_next;
        V_next = temp;
    }


    // return average concentartion of V
    double avgV = 0.0;
    for (int i = 0; i < size * size; i++) {
        avgV += V[i];
    }
    avgV /= (size * size);

    // Cleanup
    free(U);
    free(V);
    free(U_next);
    free(V_next);

    return avgV;
}


