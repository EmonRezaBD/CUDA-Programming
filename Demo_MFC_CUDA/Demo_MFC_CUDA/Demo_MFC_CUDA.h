
// Demo_MFC_CUDA.h : main header file for the PROJECT_NAME application
//

#pragma once

#ifndef __AFXWIN_H__
	#error "include 'pch.h' before including this file for PCH"
#endif

#include "resource.h"		// main symbols


// CDemoMFCCUDAApp:
// See Demo_MFC_CUDA.cpp for the implementation of this class
//

class CDemoMFCCUDAApp : public CWinApp
{
public:
	CDemoMFCCUDAApp();

// Overrides
public:
	virtual BOOL InitInstance();

// Implementation

	DECLARE_MESSAGE_MAP()
};

extern CDemoMFCCUDAApp theApp;
