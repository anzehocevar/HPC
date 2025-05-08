#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>
#include "gray_scott.h"
// Reference function for initialization of U and V
void initUV(float *U, float *V, int size)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            for (int k = 0; k < size; k++)
            {
                U[i * size * size + j * size + k] = 1.0f;
                V[i * size * size + j * size + k] = 0.0f;
            }
        }
    }
    int r = size / 8;
    for (int i = size / 2 - r; i < size / 2 + r; i++)
    {
        for (int j = size / 2 - r; j < size / 2 + r; j++)
        {
            for (int k = size / 2 - r; k < size / 2 + r; k++)
            {
                U[i * size * size + j * size + k] = 0.75f;
                V[i * size * size + j * size + k] = 0.25f;
            }
        }
    }
}

double gray_scott2D(gs_config config){
    int v = 0;
    /*
    YOUR SOLUTION GOES HERE
    Write a 2D Gray-Scott simulation in C/C++ using CUDA, OpenMPI, and OpenMP.
    */

    // return average concentartion of V
    return v;
}


