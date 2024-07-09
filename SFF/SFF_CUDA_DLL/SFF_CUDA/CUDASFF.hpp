#pragma once

//For openCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#ifndef EXP_CUDA_MATH

	#ifndef BUILD_CUDA_MATH

		#pragma comment(lib, "SFF_CUDA.lib")
		#define EXP_CUDA_MATH __declspec(dllimport)

	#else
		#define EXP_CUDA_MATH __declspec(dllexport)

	#endif

#endif

//DLL
extern "C" EXP_CUDA_MATH bool UploadImgDLL(cv::Mat img, int imgNo);//third call from DLL
extern "C" EXP_CUDA_MATH bool UploadZposDLL(double* ptr);//second call from DLL

extern "C" EXP_CUDA_MATH bool getMat(double* heightDataClient); //4th call from DLL
extern "C" EXP_CUDA_MATH int setImgCount(int imgNo); //first call from DLL


//bool UploadImgDLL(cv::Mat img, int imgNo);
//bool UploadZposDLL(std::vector<double>&vec);
//void setImgCount(int size);
//cv::Mat getMat();