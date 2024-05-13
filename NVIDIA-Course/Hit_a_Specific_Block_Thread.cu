#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<stdio.h>

__global__ void printSuccessForCorrectExecutionConfiguration()
{
    printf("threadIdx.x = %d--->blockIdx.x = %d\n", threadIdx.x, blockIdx.x);
    if (threadIdx.x == 1023 && blockIdx.x == 255) //Kernel will be executed with a grid with a total of 256 blocks.
        //Each block within the grid will contain 1024 threads
    {
        printf("Success!\n");
    }
}

int main()
{
    /*
     * This is one possible execution context that will make
     * the kernel launch print its success message.
     */

    printSuccessForCorrectExecutionConfiguration << <256, 1024 >> > ();

    /*
     * Don't forget kernel execution is asynchronous and you must
     * sync on its completion.
     */

    cudaDeviceSynchronize();
}
