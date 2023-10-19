
#include "cuda_runtime.h"
#include "device_launch_parameters.h" 

#include <stdio.h> 
#include <stdlib.h>
#include<cstring>
#include<time.h>

//S1L12:  Memory transfer between host and device
__global__ void mem_trs_test(int * input) 
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	printf("tid : %d, gid : %d, value : %d \n", threadIdx.x, gid, input[gid] );
}

__global__ void mem_trs_test2(int* input, int size) //More optimize . 
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if(gid<size)
		printf("tid : %d, gid : %d, value : %d \n", threadIdx.x, gid, input[gid]);
	//if we don't give this condition still the other 10 threads will try to acess the array and it'll find the value 0
}

int main()
{
	int size = 150;
	int byte_size = size * sizeof(int);

	int* h_input;
	h_input = (int*) malloc(byte_size);

	//time to randomly initialize the array
	time_t t;
	srand((unsigned)time(&t));
	for (int i = -0; i < size; i++)
	{
		h_input[i] = (int)(rand() & 0xff);
	}
	
	int * d_input;
	cudaMalloc((void**) &d_input, byte_size);
	cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);


	dim3 block(32);
	dim3 grid(5);
	//mem_trs_test << <grid, block >> > (d_input);
	mem_trs_test2 << <grid, block >> > (d_input, size);
	

	//thread block size affects the performance. 
	//usually we keep the thread block size to multiplication of 32

	cudaDeviceSynchronize();

	//Reclaim memory
	cudaFree(d_input);
	free(h_input);

	cudaDeviceReset();
	return 0;
}