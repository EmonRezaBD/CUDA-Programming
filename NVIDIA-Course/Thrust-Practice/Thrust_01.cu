
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//thrust headers
#include<thrust/host_vector.h>
#include<thrust/device_vector.h>

#include <iostream>
using namespace std;

/*Notes:
* = operator can also be used to copy host_vector to host_vector or device_vector to device_vector
* From cpu side the device_vector's elements can be accssed and in background it calls cudaMemcpy, but no need to manage memory ..yess!
*/

int main()
{
	thrust::host_vector<int>H(4); //Storage for 4 int vals
	cout << "Before assigning values: "<<H[0] << " " << H[1] << " " << H[3] << "\n"; //All the values are automatically initialized to zero 0

	// initialize individual elements
	H[0] = 14;
	H[1] = 20;
	H[2] = 38;
	H[3] = 46;

	cout << "After assigning values: "<< H[0] << " " << H[1] << " " << H[3] << "\n"; //All the values are automatically initialized to zero 0
	
	cout << "H has size: " << H.size() << "\n"; //size_t type data 
	// print contents of H
	for (int i = 0; i < H.size(); i++)
		cout << "H[" << i << "] = " << H[i] << endl;

	//resize H
	H.resize(2);
	cout << "After resizing H has size: " << H.size() << "\n";

	// Copy host_vector H to device_vector D
	thrust::device_vector<int> D = H; 

	cout << "Device vector's first element: " << D[0] << "\n";

	// elements of D can be modified
	D[0] = 99;
	D[1] = 88;

	// print contents of D
	for (int i = 0; i < D.size(); i++)
		std::cout << "D[" << i << "] = " << D[i] << std::endl;
	// H and D are automatically deleted when the function returns

	cout << "Program Ends\n";


	return 0;
}
