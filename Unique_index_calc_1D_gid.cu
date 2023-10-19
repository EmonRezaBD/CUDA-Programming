
#include "cuda_runtime.h"
#include "device_launch_parameters.h" 

#include <stdio.h> 

//S1L10:  Unique index calculation using 2D grid
__global__ void unique_idx_calc_threadIdx(int * input) //we will transfer memory from host to this pointer in the device
{
	int tid = threadIdx.x; //unique id for each thread
	int offset = blockIdx.x * blockDim.x;
	int gid = tid + offset; //gid = global id / unique id to access each element of an array 

	printf("threadIdx : %d, blockIdx.x : %d, blockDim.x : %d, gid : %d, array_val : %d \n", tid, blockIdx.x, blockDim.x, gid, input[gid]);
	//blocks will accesses randomly, ex. num 2 block will access then 3 , then 0 ...
}

int main()
{
	int array_size = 8;
	int array_byte_size = sizeof(int) * array_size;
	int h_data[] = {23,9,4,53,65,12,1,33};

	for (int i = 0; i < array_size; i++)
	{
		printf("%d ", h_data[i]);
	}

	printf("\n \n");

	int* d_data;
	cudaMalloc((void**)&d_data, array_byte_size);
	cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);

	dim3 block(4);//per block 4 threads
	dim3 grid(2); //there is 2 blocks
	//now in this case, first four vals will access twice by the two block of 4 threads
	//challenge : how can we access other four elements??
	//use gid
	// gid = tid + Offset
	// offset = blockidx.x * blockdim.x
	
	unique_idx_calc_threadIdx << <grid, block >> > (d_data);
	cudaDeviceSynchronize();

	cudaDeviceReset();


	return 0;
}