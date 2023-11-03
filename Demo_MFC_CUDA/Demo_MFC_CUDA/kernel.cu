#include "cudaKernel.h"

#include<stdio.h>
#include <stdlib.h>
#include<time.h>
#include <iostream>
#include <fstream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void sum_array_gpu(int* a, int* b, int* c, int size)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < size)
	{
		c[gid] = a[gid] + b[gid];
	}
}


CcudaKernel::CcudaKernel(void)
{
	success = false;
}

CcudaKernel::~CcudaKernel(void)
{
}

void CcudaKernel::DoKernel(dim3 grid, dim3 block, int* d_a, int* d_b, int* d_c, int size)
{
	sum_array_gpu << <grid, block >> > (d_a, d_b, d_c, size);//launching the kernel. As kernel launch dosen't retrun anything so we don't need to pass this for error checking
	cudaDeviceSynchronize();
}

void CcudaKernel::runTest() //like main function in CUDA
{
	int size = 10000;
	int block_size = 128;

	//cudaError error;

	int NO_BYTES = size * sizeof(int);

	//host pointers
	int* h_a, * h_b, * gpu_results, * h_c;

	//allocate memory for host pointers
	h_a = (int*)malloc(NO_BYTES);

	//h_a = new int[NO_BYTES];
	//h_a[0] = 10;
	//h_a[1] = 20;


	h_b = (int*)malloc(NO_BYTES);
	gpu_results = (int*)malloc(NO_BYTES); //this array is for holding gpu returned result
	h_c = (int*)malloc(NO_BYTES);

	time_t t;
	srand((unsigned)time(&t));

	//std::ofstream myfile;
	//myfile.open("Test.txt");
	for (int i = 0; i < size; i++)
	{
		h_a[i] = (int)(rand() & 0xFF);
		//myfile << i << " " << h_a[i] << "\n";
	}
	//myfile.close();

	for (int i = 0; i < size; i++)
	{
		h_b[i] = (int)(rand() & 0xFF);
	}

	int* d_a, * d_b, * d_c;
	//error = cudaMalloc((int**)&d_a, NO_BYTES);
	/*if (error != cudaSuccess)
	{
		fprintf(stderr, "Error : %s \n", cudaGetErrorString(error)); //but writing like this will have a big amount of code. So, we use cuda_common.cuh file and a macro to check the error.
	}*/

	cudaMalloc((int**)&d_a, NO_BYTES); //gpuErrchk is a macro..defined in other file (.cuh file),  .cuh file is a header file that contains CUDA C++ code. CUDA C++ is an extension of C++ that allows you to write code that runs on NVIDIA GPUs. .cuh files are typically used to declare functions, variables, and types that are used in CUDA C++ code.
	cudaMalloc((int**)&d_b, NO_BYTES);
	cudaMalloc((int**)&d_c, NO_BYTES); //this will populate from the kernel

	cudaMemcpy(d_a, h_a, NO_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, NO_BYTES, cudaMemcpyHostToDevice);


	//launching the grid
	dim3 block(block_size);
	dim3 grid(size / block.x + 1); //+1 will gurantee that we'll have more thread than the array size.


	DoKernel(grid, block, d_a, d_b, d_c, size); //Lunching the kernel

	//memory transfer back to host
	cudaMemcpy(gpu_results, d_c, NO_BYTES, cudaMemcpyDeviceToHost);

	std::ofstream myfile;
	myfile.open("Test.txt"); //Printing the values for seeing the output
	myfile << "Summation Result:\n ";
	for (int i = 0; i < size; i++)
	{
		myfile << i << " " << gpu_results[i] << "\n";

	}
	myfile.close();


	cudaFree(d_c);
	cudaFree(d_b);
	cudaFree(d_a);

	free(h_a);
	free(h_b);
	free(gpu_results);
}
