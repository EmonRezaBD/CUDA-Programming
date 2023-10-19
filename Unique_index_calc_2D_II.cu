
#include "cuda_runtime.h"
#include "device_launch_parameters.h" 

#include <stdio.h> 

//S1L10:  Unique index calculation using 2D grid
__global__ void unique_idx_calc_2D_II(int * input) //we will transfer memory from host to this pointer in the device
{	//2D grid with 2D blocks

	int tid = blockDim.x * threadIdx.y + threadIdx.x; 

	int num_threads_in_a_block = blockDim.x * blockDim.y;
	int block_offset = blockIdx.x * num_threads_in_a_block;

	int num_threads_in_a_row = num_threads_in_a_block * gridDim.x;
	int row_offset = num_threads_in_a_row * blockIdx.y;

	int gid = tid + block_offset + row_offset; 
	printf("threadIdx : %d, blockIdx.x : %d, blockIdx.y : %d,  gid : %d, array_val : %d \n", tid, blockIdx.x, blockIdx.y, gid, input[gid]);
	
}

int main()
{
	int array_size = 16;
	int array_byte_size = sizeof(int) * array_size;
	int h_data[] = {23,9,4,53,65,12,1,33,10,20,3,4,67,-5,-7,9};

	int* d_data;
	cudaMalloc((void**)&d_data, array_byte_size);
	cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);

	dim3 block(2,2);//per block 4 threads
	dim3 grid(2,2); //there is 2*2 blocks
	
	unique_idx_calc_2D_II<< <grid, block >> > (d_data);
	cudaDeviceSynchronize();

	cudaDeviceReset();


	return 0;
}