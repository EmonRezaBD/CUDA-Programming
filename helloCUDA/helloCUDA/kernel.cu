
#include "cuda_runtime.h"
#include "device_launch_parameters.h" //CUDA runtimes

#include <stdio.h> //Other headers for C++

//This program will print "Hello CUDA world" in the console.
/*
*CPU: int hello_world (int x, float y)
* GPU: __global__ void hello_world(intx, float y) //1 extra modifier needed to specificy it'll run in CUDA enabled device.
*/
/*
* step-1: First add header files
* step-2: write main function
* steo-3: write kernels
* step-4: Launch the kernels from the main function
*/
__global__ void hello_CUDA()
{
	printf("Hello CUDA world \n");
} //kernel

int main()
{
	//1 thread is running the func
	//hello_CUDA << <1, 1 >> > (); //launching the kernel. it is an asynchronous operation, that means host code don't have to wait for kernel to finish its work. 
								//So host code can run next inst as soon as the kernel launch is done.
	//5 thread is running parallel and will print 5 times the msg
	//hello_CUDA << <1, 5 >> > ();

	//dim3 grid(8,1,1);  //x=8, y=1,z=1 , no of thread blocks = grid
	//dim3 block(4,1,1); //x=4, y=1,z=1, no of thrads per block = block
	
	//hello_CUDA << < grid, block >> > (); //so output will be Hello CUDA world 32 times as we launched 8 thread blocks with per block 4 threads

	//2D kernel launch
	int nx, ny;
	nx = 16; //num of threads in X dimewnsion
	ny = 4; //number of threads in Y dimewnsion

	dim3 block(8, 2, 1);
	dim3 grid(nx/block.x, ny/block.y, 1);
	hello_CUDA << < grid, block >> > ();

	

	cudaDeviceSynchronize(); // now the host will wait for kernel to finish
	cudaDeviceReset(); //t is important to note that cudaDeviceReset() will interrupt all running kernels on the device. This means that you should only call cudaDeviceReset() when absolutely necessary.

}//host code

