#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<iostream>

void init(int* a, int N)
{
    int i;
    for (i = 0; i < N; ++i)
    {
        a[i] = i;
    }
}

/*
_global__ void saxpy(int n, float a, float *x, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) 
        y[i] = a * x[i] + y[i];
}
monolithic kernel: Because it assumes a single large grid of threads to process the entire array in one pass. Common CUDA guidance is to launch one thread per data element, 
which means to parallelize the above loop we write a kernel that assumes we have enough threads to more than cover the array size.
*/

/*
Grid-stride loop: Rather than assume that the thread grid is large enough to cover the entire data array, this kernel loops over the data array one grid-size at a time.
__global__ void saxpy(int n, float a, float *x, float *y)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n; 
         i += blockDim.x * gridDim.x) //blockDim.x * gridDim.x is the total number of threads in the grid
      {
          y[i] = a * x[i] + y[i];
      }
}


*/

/*
 * In the current application, `N` is larger than the grid.
 * Refactor this kernel to use a grid-stride loop in order that
 * each parallel thread work on more than one element of the array.
 */

__global__ void doubleElements(int* a, int N)
{
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    printf("blockIdx.x: %d blockDim.x: %d threadIdx.x:%d IDX: %d  gridDim.x: %d blockDim.x: %d stride:%d\n", blockIdx.x, blockDim.x, threadIdx.x, idx, gridDim.x, blockDim.x, stride);

   // printf("gridDim: %d  blockDim: %d stride: %d\n",gridDim.x, blockDim.x, stride);

    for (int i = idx; i < N; i += stride)
    {
        printf("i: %d i+Stride: %d\n", i, i += stride);
        a[i] *= 2;
    }
}

bool checkElementsAreDoubled(int* a, int N)
{
    int i;
    for (i = 0; i < N; ++i)
    {
        if (a[i] != i * 2) return false;
    }
    return true;
}

int main()
{
    /*
     * `N` is greater than the size of the grid (see below).
     */

    int N = 10000;
    int* a;

    size_t size = N * sizeof(int);
    cudaMallocManaged(&a, size);

    init(a, N);

    /*
     * The size of this grid is 256*32 = 8192.
     */

    size_t threads_per_block = 256;
    size_t number_of_blocks = 32;

    doubleElements << <number_of_blocks, threads_per_block >> > (a, N);
    cudaDeviceSynchronize();

    bool areDoubled = checkElementsAreDoubled(a, N);
    printf("All elements were doubled? %s\n", areDoubled ? "TRUE" : "FALSE");

    cudaFree(a);
}

