#include<stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//Warp details 
//Some threads are remaining idle
//Tried to use cout , but didn't work. we should use C I/O functions 
__global__ void print_details_of_warps()
{
	int gid = blockIdx.y * gridDim.x * blockDim.x +
		blockIdx.x * blockDim.x + threadIdx.x;
	int warp_id = threadIdx.x / 32;

	int gbid = blockIdx.x * gridDim.x + blockIdx.x;

	printf("tid : %d, bid.x: %d, bid.y: %d, gid: %d, warp_id: %d, gbid: %d\n",
		threadIdx.x, blockIdx.x, blockIdx.y, gid, warp_id, gbid);
}

int main(int argc, char** argv)
{
	dim3 block_size(42);
	dim3 grid_size(2, 2);

	print_details_of_warps << <grid_size, block_size>> > ();

	cudaDeviceSynchronize();
	return EXIT_SUCCESS;
}