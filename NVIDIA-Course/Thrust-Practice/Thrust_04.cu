
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//thrust headers
#include<thrust/device_vector.h>
#include <thrust/copy.h> //For copying elements
#include<list>
#include<vector>

#include <iostream>
using namespace std;

/*Notes:
* This codes covered iterators
* these are simple version
* STL like 
* Thrust also provides a collection of fancy iterators with names like counting_iteratorand zip_iterator.
*/

int main()
{
	// create an STL list with 4 values
	list<int>stl_list;

	stl_list.push_back(10);
	stl_list.push_back(20);
	stl_list.push_back(30);
	stl_list.push_back(40);

	// initialize a device_vector with the list
	thrust::device_vector<int> D(stl_list.begin(), stl_list.end());
	cout << "Before copy: "<<D[0] << "\n";

	// copy a device_vector into an STL vector
	vector<int> stl_vector(D.size());
	thrust::copy(D.begin(), D.end(), stl_vector.begin());

	cout << "After copy: " << stl_vector[3] << "\n";
	cout << "Program Ends\n";


	return 0;
}
