#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<stdio.h>
#include <stdio.h>

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
    printf("This is iteration number Block Dim: %d--->Block id: %d--->Thread Id: %d\n", blockDim.x, blockIdx.x, threadIdx.x);
    //So make it like normal for loop
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    printf("In parallel Loop %d\n", idx);

}

int main()
{
    /*
     * When refactoring `loop` to launch as a kernel, be sure
     * to use the execution configuration to control how many
     * "iterations" to perform.
     *
     * For this exercise, be sure to use more than 1 block in
     * the execution configuration.
     */

    int N = 10;
    loop<<<2,5>>>(N);
}

