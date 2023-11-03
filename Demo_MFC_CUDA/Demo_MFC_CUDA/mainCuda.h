#pragma once

// includes, system
#include <iostream>
#include <stdlib.h>

#include "cudaKernel.h"


class CmainCuda
{
public:
	 CmainCuda(void);
	~CmainCuda(void);
	// execute CUDA
	void runCuda();

public:
	CcudaKernel m_Cuda;
	cudaDeviceProp deviceProp;
	int cudartVersion;
	int driverVersion;
	int runtimeVersion;

	bool status;
	int2 i2[16];
	CString inpstr;
	CString outstr;
	CString success;


};

