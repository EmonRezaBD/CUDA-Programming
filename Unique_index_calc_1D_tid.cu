
#include "cuda_runtime.h"
#include "device_launch_parameters.h" 

#include <stdio.h> 

//S1L9:  Unique index calculation
__global__ void unique_idx_calc_threadIdx(int * input) //we will transfer memory from host to this pointer in the device
{
	int tid = threadIdx.x; //unique id for thread 
	printf("threadIdx : %d, value : %d \n", tid, input[tid]);

}

int main()
{
	int array_size = 8;
	int array_byte_size = sizeof(int) * array_size;
	int h_data[] = {23,9,4,53,65,12,1,33}; //Allocating memory in the host code

	for (int i = 0; i < array_size; i++)
	{
		printf("%d ", h_data[i]);
	}

	printf("\n \n");

	int* d_data;
	cudaMalloc((void**)&d_data, array_byte_size); //allocating memory in GPU device
	cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice); //copy data from host to GPU : there is other mode as well. see the def of the cudaMemcpy()

	dim3 block(8);
	dim3 grid(1);
	
	unique_idx_calc_threadIdx << <grid, block >> > (d_data); //calculate with the kernel
	cudaDeviceSynchronize(); //After this command output will show in the console

	cudaDeviceReset();


	return 0;
}