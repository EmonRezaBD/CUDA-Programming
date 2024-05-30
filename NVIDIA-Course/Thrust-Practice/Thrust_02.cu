
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
* The copy function can be used to copy a range of host or device elements to another host or device vector
* thrust::fill simply sets a range of elements to a specific value
* Thrust’s sequence function can be used to create a sequence of equally spaced values
*/

int main()
{
	// initialize all ten integers of a device_vector to 1
	thrust::device_vector<int>D(10, 1);
	cout << "Before fill: " << D[5] << "\n";

	// set the first seven elements of a vector to 9
	thrust::fill(D.begin(), D.begin() + 7, 9);
	cout << "After fill: "<<D[5] << "\n";

	// initialize a host_vector with the first five elements of D
	thrust::host_vector<int> H(D.begin(), D.begin() + 5);
	cout << "After fill host vec: " << H[4] << "\n";

	// set the elements of H to 0, 1, 2, 3, ...
	thrust::sequence(H.begin(), H.end());
	cout << "After sequence host vec: " << H[4] << "\n";

	// copy all of H back to the beginning of D
	thrust::copy(H.begin(), H.end(), D.begin());
	cout << "After copy device vec: " << D[0] <<" "<<D[3] << "\n";

	// print D
	for (int i = 0; i < D.size(); i++)
		std::cout << "D[" << i << "] = " << D[i] << std::endl;


	cout << "Program Ends\n";


	return 0;
}
