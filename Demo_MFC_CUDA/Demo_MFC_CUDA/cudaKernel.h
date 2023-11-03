//This file is for storing all the functions use in .cu file
#pragma once

#include <stdlib.h>
#include <stdio.h>

// includes, project
//#include <cutil_inline.h>

// Required header to support CUDA vector types
#include <vector_types.h>
#include <cufft.h>

class CcudaKernel
{
public:
	 CcudaKernel(void);
	~CcudaKernel(void);

	void DoKernel(dim3 grid, dim3 block, int* d_a, int* d_b, int* d_c, int);
	void runTest();

public:
	bool success;

};
