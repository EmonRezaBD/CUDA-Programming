
#include "cuda_runtime.h"
#include "device_launch_parameters.h" 
//#include "../cuda_common.cuh"
#include "../common.h"
#include <stdio.h> 
#include <stdlib.h>
#include<time.h>

//for memset
#include<cstring>

//Section 1 L 14
//Sum array with validity check
//Idea: first thread will add arr1[0]+arr2[0] and store the result in sum[0]
//Issue: we don't have any way to confirm the results of calculation in GPU. So, to verify the result of GPU implementation, we need
//to compare it with CPU implementation. 
//But we need to place the vilidity check code in a common place. 

__global__ void sum_array_gpu(int* a, int* b, int* c, int size)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (gid < size) 
	{
		c[gid] = a[gid] + b[gid];
	}
}

void sum_array_cpu(int* a, int* b, int* c, int size)
{
	for (int i = 0; i < size; i++)
	{
		c[i] = a[i] + b[i];
	}
}

int main()
{
	int size = 10000;
	int block_size = 128;

	int NO_BYTES = size * sizeof(int);

	//host pointers
	int * h_a, * h_b, * gpu_results, *h_c;

	//allocate memory for host pointers
	h_a = (int*)malloc(NO_BYTES);
	h_b = (int*)malloc(NO_BYTES);
	gpu_results = (int*)malloc(NO_BYTES); //this array is for holding gpu returned result
	h_c = (int*)malloc(NO_BYTES);

	time_t t;
	srand((unsigned)time(&t));

	for (int i = 0; i < size; i++)
	{
		h_a[i] = (int)(rand()& 0xFF);
	}

	for (int i = 0; i < size; i++)
	{
		h_b[i] = (int)(rand() & 0xFF);
	}

	sum_array_cpu(h_a, h_b, h_c, size);
	//device pointer
	int* d_a, * d_b, * d_c;
	cudaMalloc((int**)&d_a, NO_BYTES);
	cudaMalloc((int**)&d_b, NO_BYTES);
	cudaMalloc((int**)&d_c, NO_BYTES); //this will populate from the kernel

	cudaMemcpy(d_a, h_a, NO_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, NO_BYTES, cudaMemcpyHostToDevice);

	//launching the grid
	dim3 block(block_size);
	dim3 grid(size/block.x+1); //+1 will gurantee that we'll have more thread than the array size.

	sum_array_gpu << <grid, block >> > (d_a, d_b, d_c, size) ;
	cudaDeviceSynchronize();

	//memory transfer back to host
	cudaMemcpy(gpu_results, d_c, NO_BYTES, cudaMemcpyDeviceToHost);

	//array comparison
	compare_arrays(gpu_results, h_c, size);

	cudaFree(d_c);
	cudaFree(d_b);
	cudaFree(d_a);

	free(h_a);
	free(h_b);
	free(gpu_results);

	return 0;
}