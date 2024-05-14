#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<stdio.h>
#include<cstdlib>
#include<assert.h>

#define N  64

__global__ void matrixMulGPU(int* a, int* b, int* c)
{
    /*
     * Build out this kernel.
     */
    int val = 0;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    printf("blockIdx.x %d  blockDim.x %d threadIdx.x %d row: %d\n", blockIdx.x, blockDim.x, threadIdx.x, row);
    //printf("blockIdx.y %d  blockDim.y %d threadIdx.y %d col: %d\n", blockIdx.y, blockDim.y, threadIdx.y, col);


    if (row < N && col < N)
    {
        for (int k = 0; k < N; k++)
        {
            val += a[row * N + k] * b[k * N + col];
           //printf("a[i]:  %d  b[i]: %d\n", row * N + k, k * N + col); //Multiple values of row * N + k, because matrix mul formula
          // printf("a[%d]: %d b[%d]: %d\n", row * N + k, a[row * N + k], k * N + col, b[k * N + col]);
        }
        c[row * N + col] = val;
    }
}

/*
 * This CPU function already works, and will run to create a solution matrix
 * against which to verify your work building out the matrixMulGPU kernel.
 */

void matrixMulCPU(int* a, int* b, int* c)
{
    int val = 0;

    for (int row = 0; row < N; ++row)
        for (int col = 0; col < N; ++col)
        {
            val = 0;
            for (int k = 0; k < N; ++k)
                val += a[row * N + k] * b[k * N + col];
            c[row * N + col] = val;
        }
}

int main()
{
    int* a, * b, * c_cpu, * c_gpu; // Allocate a solution matrix for both the CPU and the GPU operations

    int size = N * N * sizeof(int); // Number of bytes of an N x N matrix

    // Allocate memory
    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c_cpu, size);
    cudaMallocManaged(&c_gpu, size);

    // Initialize memory; create 2D matrices
    for (int row = 0; row < N; ++row)
        for (int col = 0; col < N; ++col)
        {
            a[row * N + col] = row;

            b[row * N + col] = col + 2;
            //printf("a[%d]: %d b[%d]: %d\n", row * N + col, a[row * N + col], row * N + col, b[row * N + col]);

            c_cpu[row * N + col] = 0;
            c_gpu[row * N + col] = 0;
        }

    /*
     * Assign `threads_per_block` and `number_of_blocks` 2D values
     * that can be used in matrixMulGPU above.
     */

    dim3 threads_per_block (16,16,1);
    dim3 number_of_blocks ( (N/threads_per_block.x)+1, (N/threads_per_block.y)+1, 1);
    /*For N = 64 and threads_per_block.x = 16, (64/16) + 1 = 4 + 1 = 5.
Similarly, for the y-dimension, (64/16) + 1 = 5.
Therefore, the grid consists of 5 x 5 blocks.
    Total threads = Number of blocks in x-dimension * Number of blocks in y-dimension * Threads per block
               = 5 * 5 * 256
               = 6400 threads
     */

    matrixMulGPU << < number_of_blocks, threads_per_block >> > (a, b, c_gpu);

    cudaDeviceSynchronize();

    // Call the CPU version to check our work
    matrixMulCPU(a, b, c_cpu);

    // Compare the two answers to make sure they are equal
    bool error = false;
    for (int row = 0; row < N && !error; ++row)
        for (int col = 0; col < N && !error; ++col)
            if (c_cpu[row * N + col] != c_gpu[row * N + col])
            {
                printf("FOUND ERROR at c[%d][%d]\n", row, col);
                error = true;
                break;
            }
    if (!error)
        printf("Success!\n");

    // Free all our allocated memory
    cudaFree(a); cudaFree(b);
    cudaFree(c_cpu); cudaFree(c_gpu);
}

