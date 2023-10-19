
#include "cuda_runtime.h"
#include "device_launch_parameters.h" 

#include <stdio.h> 

//S1L10:  Unique index calculation using 2D grid
__global__ void unique_idx_calc_2D(int * input) //we will transfer memory from host to this pointer in the device
{
	int tid = threadIdx.x; //unique id for each thread
	int block_offset = blockIdx.x * blockDim.x;

	int row_offset = blockDim.x * gridDim.x * blockIdx.y;

	int gid = tid + block_offset + row_offset; 
	printf("threadIdx : %d, blockIdx.x : %d, blockIdx.y : %d,  gid : %d, array_val : %d \n", tid, blockIdx.x, blockIdx.y, gid, input[gid]);
	//blocks will accesses randomly, ex. num 2 block will access then 3 , then 0 ...
}

int main()
{
	int array_size = 16;
	int array_byte_size = sizeof(int) * array_size;
	int h_data[] = {23,9,4,53,65,12,1,33,10,20,3,4,67,-5,-7,9};

	int* d_data;
	cudaMalloc((void**)&d_data, array_byte_size);
	cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);

	dim3 block(4);//per block 4 threads
	dim3 grid(2,2); //there is 2*2 blocks

	//So total value needed: 2*2*4 = 16
	//if we use this eqn int gid = tid + offset, in case of 2d it'll not work . we have  to consider y axis as well
	//So for 2D: gid/index = (tid  + block offset) + row offset
	//					   =(threadIdx.x+ number of threads in thread block * blockIdx.x) + number of threads in one thread block row * blockidx.y
	//					   number of threads in one row = gridDim.x * blockDim.x
	//					   number of threads in thread block = blockDim.x
	// SO final eqn
	// gid = threadidx.x +  blockDim.x * blockIdx.x +  gridDim.x * blockDim.x * blockIdx.y
	// row offset = gridDim.x * blockDim.x * blockIdx.y (for y dimension)
	// block offset =  blockDim.x * blockIdx.x (for x dimension)
	
	unique_idx_calc_2D<< <grid, block >> > (d_data);
	cudaDeviceSynchronize();

	cudaDeviceReset();


	return 0;
}