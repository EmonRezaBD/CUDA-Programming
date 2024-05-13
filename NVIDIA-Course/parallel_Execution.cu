#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

/*
 * Refactor firstParallel so that it can run on the GPU.
 */

__global__ void firstParallel()
{
	int i = threadIdx.x;
	printf("%d This should be running in parallel.\n", i);
}

int main()
{
	/*
	 * Refactor this call to firstParallel to execute in parallel
	 * on the GPU.
	 */

	firstParallel<<<5,5>>>();
	cudaDeviceSynchronize();

	/*
	 * Some code is needed below so that the CPU will wait
	 * for the GPU kernels to complete before proceeding.
	 */

}
