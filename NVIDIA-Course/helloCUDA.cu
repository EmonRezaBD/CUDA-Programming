
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

void helloCPU()
{
	printf("Hello from the CPU\n");
}

__global__ void helloGPU()
{
	printf("Hello from the GPU\n");
}

int main()
{
	helloCPU();
	
	//launch the kernel
	helloGPU <<< 1, 5 >>> (); //will print total 5 times
	cudaDeviceSynchronize();
	
	return 0;
}