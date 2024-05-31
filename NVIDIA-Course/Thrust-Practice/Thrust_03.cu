
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//thrust headers
#include<thrust/host_vector.h>
#include<thrust/device_vector.h>

#include <thrust/copy.h> //For copying elements
#include <thrust/fill.h> //For setting with a val
#include <thrust/sequence.h> //For fill up with a sequence

#include <iostream>
using namespace std;

/*Notes:
* Be carful with the namespaces, For instance, thrust::copy is different from std::copy 
* Iterators and Static Dispatching:  no runtime overhead for function call
* We can also work with raw pointers
* need to wrap it with thrust::device_ptr before calling the function
* 
*/

int main()
{
	size_t N = 10;

	// raw pointer to device memory
	int* raw_ptr;
	cudaMalloc((void**)&raw_ptr, N * sizeof(int));

	// wrap raw pointer with a device_ptr 
	thrust::device_ptr<int> dev_ptr(raw_ptr);

	// use device_ptr in thrust algorithms
	thrust::fill(dev_ptr, dev_ptr + N, (int)0 );

	//To extract a raw pointer from a device_ptr the raw_pointer_cast should be applied as follows :
	//thrust::device_ptr<int> dev_ptr = thrust::device_malloc<int>(N); //Not working


	// extract raw pointer from device_ptr
	int* raw_ptr1 = thrust::raw_pointer_cast(dev_ptr);
	//cout << dev_ptr[0];

	cout << "Program Ends\n";


	return 0;
}
