#include <cuda_runtime.h> // Include the CUDA runtime header
#include "device_launch_parameters.h" 

#include <iostream>
#include <limits.h>
#include<cstdlib>
#include <time.h>
#include<cassert>

using namespace std;

__global__ void matrixMul(int *a, int *b, int *c, int N)
{
	//Calculate the global row and column for each thread

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int column = blockIdx.x * blockDim.x + threadIdx.x;

	//Boundary check for our matrix
	if (row < N && column < N)
	{
		int tmp = 0;
		for (int i = 0; i < N; i++)
		{
			tmp += a[row * N + i] * b[i * N + column];
		}
		//Write back the result
		c[row * N + column] = tmp;
	}

}

void init_matrix(int * m, int N) //0-100, square mat 
{
	for (int i = 0; i < N * N; i++)
	{
		m[i] = rand() % 100;
	}
}

//Verify the result on the CPU
void verify_result(int* a, int* b, int* c, int N)
{
	int tmp;
	//For every row
	for (int i = 0; i < N; i++)
	{   
		//For every column
		for (int j = 0; j < N; j++)
		{
			//For every element int the row-col pair
			tmp = 0;
			for (int k = 0; k < N; k++)
			{
				tmp += a[i * N + k] * b[k * N + j];
			}
			//Check each result 
			assert(tmp == c[i * N + j]);
		}
	}
}

int main()
{
	int N = 1 << 10; //1024x1024
	size_t bytes = N * N * sizeof(int);

	time_t mem_start, mem_end;
	mem_start = clock();
	//Allocate memory
	int* a, * b, * c;
	cudaMallocManaged(&a, bytes); //input mat
	cudaMallocManaged(&b, bytes); //input mat
	cudaMallocManaged(&c, bytes); //output mat
	mem_end = clock();
	printf("Memory trnasfer time : %4.6f \n",
		(double)((double)(mem_end - mem_start) / CLOCKS_PER_SEC));

	//Initialize the data
	init_matrix(a, N);
	init_matrix(b, N);

	//Set our thread and block
	int threads = 16;
	int blocks = (N + threads - 1) / threads;

	//setup our kernel lunch parameters
	dim3 THREADS(threads, threads);
	dim3 BLOCKS(blocks, blocks);

	time_t kernel_start, kernel_end;
	kernel_start = clock();
	//Launch our kernel
	matrixMul << <BLOCKS, THREADS >> > (a,b,c,N);
	cudaDeviceSynchronize();
	kernel_end = clock();
	printf("Kernel multiplication time : %4.6f \n",
		(double)((double)(kernel_end - kernel_start) / CLOCKS_PER_SEC));

	time_t cpu_start, cpu_end;
	cpu_start = clock();
	//Verify the resut
	verify_result(a, b, c, N);
	cpu_end = clock();
	printf("CPU multiplication time : %4.6f \n",
		(double)((double)(cpu_end - cpu_start) / CLOCKS_PER_SEC));

	cout << "Program completed\n";

	return 0;
}
