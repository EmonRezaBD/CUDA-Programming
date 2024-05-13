#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<stdio.h>

/*
 * Refactor `loop` to be a CUDA Kernel. The new kernel should
 * only do the work of 1 iteration of the original loop.
 */

__global__ void loop(int N)
{
    /*for (int i = 0; i < N; ++i)
    {
        printf("This is iteration number %d\n", i);
    }*/

    printf("This is iteration number %d\n", threadIdx.x);

}

int main()
{
    /*
     * When refactoring `loop` to launch as a kernel, be sure
     * to use the execution configuration to control how many
     * "iterations" to perform.
     *
     * For this exercise, only use 1 block of threads.
     */

    int N = 10;

    loop<<<1,10>>>(N);
    cudaDeviceSynchronize();
}

