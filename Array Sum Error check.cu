
#include "cuda_runtime.h"
#include "device_launch_parameters.h" 
#include "../cuda_common.cuh"
#include "../common.h"
#include <stdio.h> 
#include <stdlib.h>
#include<time.h>

//for memset
#include<cstring>

//Section 1 L 15
//Sum array with error check
//Idea: we have separate 02 devices so we need a way to trnasfer error happened in device to host.  Use cudaError cuda_function() and cudaGetErrorString(error)
//***Important: Use error check in every cuda function calls for industrial development
//We are now only checking the error but in practical application, we have do define other paths for the execution

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

	cudaError error;

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
	//error = cudaMalloc((int**)&d_a, NO_BYTES);
	/*if (error != cudaSuccess)
	{
		fprintf(stderr, "Error : %s \n", cudaGetErrorString(error)); //but writing like this will have a big amount of code. So, we use cuda_common.cuh file and a macro to check the error. 
	}*/

	gpuErrchk(cudaMalloc((int**)&d_a, NO_BYTES)); //gpuErrchk is a macro..defined in other file (.cuh file),  .cuh file is a header file that contains CUDA C++ code. CUDA C++ is an extension of C++ that allows you to write code that runs on NVIDIA GPUs. .cuh files are typically used to declare functions, variables, and types that are used in CUDA C++ code.
	gpuErrchk(cudaMalloc((int**)&d_b, NO_BYTES));
	gpuErrchk(cudaMalloc((int**)&d_c, NO_BYTES)); //this will populate from the kernel

	gpuErrchk(cudaMemcpy(d_a, h_a, NO_BYTES, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_b, h_b, NO_BYTES, cudaMemcpyHostToDevice));

	//launching the grid
	dim3 block(block_size);
	dim3 grid(size/block.x+1); //+1 will gurantee that we'll have more thread than the array size.

	sum_array_gpu << <grid, block >> > (d_a, d_b, d_c, size) ;//launching the kernel. As kernel launch dosen't retrun anything so we don't need to pass this for error checking
	cudaDeviceSynchronize();

	//memory transfer back to host
	gpuErrchk(cudaMemcpy(gpu_results, d_c, NO_BYTES, cudaMemcpyDeviceToHost));

	//array comparison
	compare_arrays(gpu_results, h_c, size);

	gpuErrchk(cudaFree(d_c));
	gpuErrchk(cudaFree(d_b));
	gpuErrchk(cudaFree(d_a));

	free(h_a);
	free(h_b);
	free(gpu_results);

	return 0;
}