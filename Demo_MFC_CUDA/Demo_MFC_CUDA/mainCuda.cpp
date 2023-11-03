#include "pch.h"
#include "mainCuda.h"

CmainCuda::CmainCuda(void)
{
	status = false;
}

CmainCuda::~CmainCuda(void)
{

}
// execute CUDA
void CmainCuda::runCuda()
{
	m_Cuda.runTest(); //To run cuda functions 
}


