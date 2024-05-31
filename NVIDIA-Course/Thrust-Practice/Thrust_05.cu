
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//thrust headers
#include<thrust/device_vector.h>
#include<thrust/transform.h>
#include<thrust/sequence.h>

#include <thrust/copy.h> //For copying elements
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>

#include <iostream>
using namespace std;

/*Notes:
* We have parallel algorithms: thrust::sort and std::sort
* 
*/

int main()
{
	// allocate three device_vectors with 10 elements
	thrust::device_vector<int> X(10);
	thrust::device_vector<int> Y(10);
	thrust::device_vector<int> Z(10);

	// initialize X to 0,1,2,3, ....
	thrust::sequence(X.begin(), X.end());

	// compute Y = -X
	thrust::transform(X.begin(), X.end(), Y.begin(), thrust::negate<int>());
	cout << "X: "<< X[0] <<" Y: "<<Y[0] << "\n";

	for (int i = 0; i < Y.size(); ++i)
	{
		cout << "Y[" << i << "] = " << Y[i] << "\n";
	}

	// fill Z with twos
	thrust::fill(Z.begin(), Z.end(), 2);

	//compute Y = X mod 2
	thrust::transform(X.begin(), X.end(), Z.begin(), Y.begin(), thrust::modulus<int>()); //doing modulus operation in each element of X and assigning it to Y
	for (int i = 0; i < Y.size(); ++i)
	{
		cout << "Y[" << i << "] = " << Y[i] << "\n";
	}

	// replace all the ones in Y with tens
	thrust::replace(Y.begin(), Y.end(), 1, 10); //Searching 1 and replacing it with 10


	// print Y
	thrust::copy(Y.begin(), Y.end(), std::ostream_iterator<int>(std::cout, "\n"));//printing each element like for loop i++
	 
	cout << "Program Ends\n";

	return 0;
}
