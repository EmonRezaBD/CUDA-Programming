////
////#include "cuda_runtime.h"
////#include "device_launch_parameters.h"
////
/////* openCVGrab: A sample program showing to convert Pylon images to opencv MAT.
////
////	Copyright 2017 Matthew Breit <matt.breit@gmail.com>
////
////	THIS SOFTWARE REQUIRES ADDITIONAL SOFTWARE (IE: LIBRARIES) IN ORDER TO COMPILE
////	INTO BINARY FORM AND TO FUNCTION IN BINARY FORM. ANY SUCH ADDITIONAL SOFTWARE
////	IS OUTSIDE THE SCOPE OF THIS LICENSE.
////*/
//// Include files to use OpenCV API.
////#include <opencv2/core/core.hpp>
////#include <opencv2/highgui/highgui.hpp>
////#include <opencv2/core/cuda.hpp>
////
////#include <opencv2/cudaimgproc.hpp> //for filtering
////#include <opencv2/cudafilters.hpp>  //for filtering
////#include <opencv2/cudaarithm.hpp> //for abs
////#include <opencv2/imgcodecs.hpp>     // Image file reading and writing
////
////
//// Include files to use the PYLON API.
////#include <pylon/PylonIncludes.h>
////
//// Use sstream to create image names including integer
////#include <iostream>
////#include <stdio.h>
////#include <fstream>
////#include <vector>
////#include <string>
////#include <sstream>
////
//// Namespace for using pylon objects.
////using namespace Pylon;
////
//// Namespace for using GenApi objects
////using namespace GenApi;
////
////std
////using namespace std;
////
//// Number of images to be grabbed.
////static const uint32_t c_countOfImagesToGrab = 10;
////
////void printMat(cv::Mat img)
////{
////	std::ofstream file;
////	file.open("D:\\CUDA_WLI\\Data Acquisition\\openCVImage.csv");
////
////	if (!file.is_open()) {
////		std::cerr << "Failed to open the file!" << std::endl;
////		return;
////	}
////	 Iterate over each pixel and write to the file
////	for (int i = 0; i < img.rows; ++i) {
////		for (int j = 0; j < img.cols; ++j) {
////			 Get pixel value
////			uchar pixel = img.at<uchar>(i, j);
////			file << img.at<float>(i, j);
////			file << img.at<uchar>(i, j);
////			if (j != img.cols - 1) file << ", ";  // Avoid comma at the end of the line
////		}
////		file << "\n";  // Newline for the next row
////	}
////
////	 Close the file
////	file.close();
////
////	std::cout << "Image data written to CSV file successfully." << std::endl;
////}
////
////int main(int argc, char* argv[])
////{
////	std::cout << "Program Starts\n";
////	 The exit code of the sample application.
////	int exitCode = 0;
////
////	 Automatically call PylonInitialize and PylonTerminate to ensure the pylon runtime system
////	 is initialized during the lifetime of this object.
////	Pylon::PylonAutoInitTerm autoInitTerm;
////
////	try
////	{
////		 Create an instant camera object with the camera device found first.
////		std::cout << "Creating Camera..." << std::endl;
////		CInstantCamera camera(CTlFactory::GetInstance().CreateFirstDevice()); //Getting the camera and no exception
////		 or use a device info object to use a specific camera
////		CDeviceInfo info;
////		info.SetSerialNumber("21694497");
////		CInstantCamera camera( CTlFactory::GetInstance().CreateFirstDevice(info));
////
////		std::cout << "Camera Created." << std::endl;
////		 Print the model name of the camera.
////		std::cout << "Using device " << camera.GetDeviceInfo().GetModelName() << std::endl;
////
////		 The parameter MaxNumBuffer can be used to control the count of buffers
////		 allocated for grabbing. The default value of this parameter is 10.
////		camera.MaxNumBuffer = 10;
////
////		 create pylon image format converter and pylon image
////		CImageFormatConverter formatConverter;
////		formatConverter.OutputPixelFormat = PixelType_BGR8packed;
////		CPylonImage pylonImage;
////
////		 Create an OpenCV image
////		cv::Mat openCvImage;
////
////		 Start the grabbing of c_countOfImagesToGrab images.
////	    The camera device is parameterized with a default configuration which
////	    sets up free-running continuous acquisition.
////		camera.StartGrabbing(c_countOfImagesToGrab);
////
////		 This smart pointer will receive the grab result data.
////		CGrabResultPtr ptrGrabResult;
////
////		 Camera.StopGrabbing() is called automatically by the RetrieveResult() method
////	    when c_countOfImagesToGrab images have been retrieved.
////
////		while (camera.IsGrabbing())
////		{
////			 Wait for an image and then retrieve it. A timeout of 5000 ms is used.
////			camera.RetrieveResult(5000, ptrGrabResult, TimeoutHandling_ThrowException);
////			 Image grabbed successfully?
////
////			if (ptrGrabResult->GrabSucceeded())
////			{
////				 Access the image data.
////				cout << "SizeX: " << ptrGrabResult->GetWidth() << endl;
////				cout << "SizeY: " << ptrGrabResult->GetHeight() << endl;
////				const uint8_t* pImageBuffer = (uint8_t*)ptrGrabResult->GetBuffer();
////				cout << "Gray value of first pixel: " << (uint32_t)pImageBuffer[0] << endl << endl;
////
////				 Convert the grabbed buffer to pylon image
////				formatConverter.Convert(pylonImage, ptrGrabResult);
////				 Create an OpenCV image out of pylon image
////				openCvImage = cv::Mat(ptrGrabResult->GetHeight(), ptrGrabResult->GetWidth(), CV_8UC3, (uint8_t*)pylonImage.GetBuffer());
////
////				cv::Mat gray;
////				cv::cvtColor(openCvImage, gray, cv::COLOR_BGR2GRAY);
////
////				cv::Mat floatImg;
////				floatImg.convertTo(gray, CV_32FC1);
////
////				print image
////				printMat(gray);
////
////				 Create a display window
////				cv::namedWindow("OpenCV Display Window", cv::WINDOW_NORMAL);//AUTOSIZE //FREERATIO
////
////				 Display the current image with opencv
////				cv::imshow("OpenCV Display Window", openCvImage);
////
////			}
////			else
////			{
////				cout << "Error: " << ptrGrabResult->GetErrorCode() << " " << ptrGrabResult->GetErrorDescription() << endl;
////			}
////		}
////
////
////
////
////	}
////	catch (GenICam::GenericException& e)
////	{
////		 Error handling.
////		std::cerr << "An exception occurred." << std::endl
////			<< e.GetDescription() << std::endl;
////		exitCode = 1;
////	}
////
////	 Comment the following two lines to disable waiting on exit.
////	cerr << endl << "Press Enter to exit." << endl;
////	while (cin.get() != '\n');
////
////	return exitCode;
////
////	cv::cuda::GpuMat test;
////	Pylon::PylonAutoInitTerm autoInitTerm;
////	//Pylon::PylonAutoInitTerm autoInitTerm;
////	getchar();
////	std::cout << "Program Ends\n";
////	return 0;
////}