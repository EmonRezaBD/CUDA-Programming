// Grab.cpp
/*
   This sample illustrates how to grab and process images using the CInstantCamera class.
   The images are grabbed and processed asynchronously, i.e.,
   while the application is processing a buffer, the acquisition of the next buffer is done
   in parallel.

   The CInstantCamera class uses a pool of buffers to retrieve image data
   from the camera device. Once a buffer is filled and ready,
   the buffer can be retrieved from the camera object for processing. The buffer
   and additional image data are collected in a grab result. The grab result is
   held by a smart pointer after retrieval. The buffer is automatically reused
   when explicitly released or when the smart pointer object is destroyed.
*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Include files to use the PYLON API.
#include <pylon/PylonIncludes.h>
#include <pylon/usb/BaslerUsbInstantCamera.h>   // MJR
#ifdef PYLON_WIN_BUILD
#    include <pylon/PylonGUI.h>
#endif

//#include "../include/SampleImageCreator.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/types_c.h>

//For CUDA
#include <opencv2/cudaimgproc.hpp> //for filtering
#include <opencv2/cudafilters.hpp>  //for filtering
#include <opencv2/cudaarithm.hpp> //for abs
#include <opencv2/imgcodecs.hpp>     // Image file reading and writing

#include <iostream>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

#include<chrono>
#include<thread>
//#include <unistd.h>

using namespace cv;
using namespace std;



// Namespace for using pylon objects.
using namespace Pylon;
//using namespace GenApi; // needed for image read/write


// Namespace for using cout.
using namespace std;

// Number of images to be grabbed.
static const uint32_t c_countOfImagesToGrab = 1;

clock_t tStart;
clock_t tCount;

void printMat(cv::Mat img)
{
	std::ofstream file;
	file.open("D:\\CUDA_WLI\\Data Acquisition\\FInalImg.csv");

	if (!file.is_open()) {
		std::cerr << "Failed to open the file!" << std::endl;
		return;
	}
	// Iterate over each pixel and write to the file
	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			 //Get pixel value
			//uchar pixel = img.at<uchar>(i, j);
			file << (int)img.at<char>(i, j);
			//file << (float)img.at<uchar>(i, j);
            //file << img.at<uint32_t>(i, j);
			if (j != img.cols - 1) file << ", ";  // Avoid comma at the end of the line
		}
		file << "\n";  // Newline for the next row
	}

	 //Close the file
	file.close();

	std::cout << "Image data written to CSV file successfully." << std::endl;
}

__global__ void gpuStart(float* result, float* a, int b)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    result[idx] = a[idx] + b;
}

int main(int argc, char* argv[])
{
    // The exit code of the sample application.
    int exitCode = 0;

    // Automagically call PylonInitialize and PylonTerminate to ensure the pylon runtime system
    // is initialized during the lifetime of this object.
    Pylon::PylonAutoInitTerm autoInitTerm;

    //cv::Mat theFrame;

    try
    {
        // Create an instant camera object with the camera device found first.
        //CInstantCamera camera( CTlFactory::GetInstance().CreateFirstDevice());
        CBaslerUsbInstantCamera camera(CTlFactory::GetInstance().CreateFirstDevice()); //MJR

        // Print the model name of the camera.
        cout << "Using device " << camera.GetDeviceInfo().GetModelName() << endl;

        // The parameter MaxNumBuffer can be used to control the count of buffers
        // allocated for grabbing. The default value of this parameter is 10.
        camera.MaxNumBuffer = 5;

        // Start the grabbing of c_countOfImagesToGrab images.
        // The camera device is parameterized with a default configuration which
        // sets up free-running continuous acquisition.
        camera.StartGrabbing(c_countOfImagesToGrab);

        // This smart pointer will receive the grab result data.
        CGrabResultPtr ptrGrabResult;

        // Camera.StopGrabbing() is called automatically by the RetrieveResult() method
        // when c_countOfImagesToGrab images have been retrieved.
        tStart = clock();
        while (camera.IsGrabbing())
        {
            // Wait for an image and then retrieve it. A timeout of 5000 ms is used.
            camera.RetrieveResult(5000, ptrGrabResult, TimeoutHandling_ThrowException);

            // Image grabbed successfully?
            if (ptrGrabResult->GrabSucceeded())
            {
                // Access the image data.
               // cout << "SizeX: " << ptrGrabResult->GetWidth() << endl;
               // cout << "SizeY: " << ptrGrabResult->GetHeight() << endl;
               const uint8_t* pImageBuffer = (uint8_t*)ptrGrabResult->GetBuffer();
               //// const uchar* pImageBuffer = (uchar*)ptrGrabResult->GetBuffer();

               cout << "Gray value of 32 bit first pixel: " << (uint32_t)pImageBuffer[0] << endl << endl;
               cout << "Gray value of 32 bit first pixel: " << (int)pImageBuffer[0] << endl << endl;
               cout << "Gray value of 32 bit 2nd pixel: " << (int)pImageBuffer[1] << endl << endl;
               cout << "Gray value of 32 bit last pixel: " << (int)pImageBuffer[1279] << endl << endl;
               cout << "Gray value of 32 bit second last pixel: " << (int)pImageBuffer[1024] << endl << endl;

                //cout << "Gray value of 8 bit first pixel: " << (uint8_t)pImageBuffer[0] << endl << endl; //ASCII Values

#ifdef PYLON_WIN_BUILD
                // Display the grabbed image.
               // Pylon::DisplayImage(1, ptrGrabResult);
#endif

                // // Save the image using Pylon API
                // CImagePersistence::Save( ImageFileFormat_Png, "GrabbedImage.png", ptrGrabResult);


                // suppose your camera is monochrome... get a pointer to pylon image 
                //const pylon::uint8_t *pImageBuffer = (uint8_t *)ptrGrabResult->GetBuffer();
                int frameCols = ptrGrabResult->GetWidth();
                int frameRows = ptrGrabResult->GetHeight();
                // Map the pylon image buffer to a cv::Mat (create a cv::Mat from external buffer)

                //theFrame = cv::Mat(cv::Size(frameCols, frameRows), CV_8UC1, (void*)pImageBuffer, cv::Mat::AUTO_STEP);
                //Mat theFrame = cv::Mat(cv::Size(frameCols, frameRows), CV_8UC1, (void*)pImageBuffer);
             
             /*   Mat theFrame = cv::Mat(frameRows, frameCols, CV_8UC1);
                memcpy(theFrame.ptr(), (uint8_t*)ptrGrabResult->GetBuffer(), frameCols * frameRows);*/

                Mat theFrame = cv::Mat(frameRows, frameCols, CV_8UC1);
                memcpy(theFrame.ptr(), (uint8_t*)ptrGrabResult->GetBuffer(), frameCols * frameRows);

                //Gpu Start Code
                const int N = 10;
                size_t size = N * sizeof(float);
                int b = 100;

                float* a;
                float* c;

                cudaMallocManaged(&a, size);
                cudaMallocManaged(&c, size);

                for (int i = 0; i < N; ++i)
                {
                    a[i] = i + 1;
                }
     
                //gpuStart << <1, 10 >> > (c, a, b);

                cv::cuda::GpuMat gpuImg;

                clock_t cpu_start, cpu_end;
                cpu_start = clock();
                gpuImg.upload(theFrame);
                cpu_end = clock();

                printf(" Single image transfer from host to device : %4.6f \n",
                    (double)((double)(cpu_end - cpu_start) / CLOCKS_PER_SEC));


                cv::imshow("Fig", theFrame);
                cv::waitKey(0);
                printMat(theFrame);

                //cv::Mat gray;
                //cv::cvtColor(theFrame, gray, cv::COLOR_BGR2GRAY);
                //cv::Mat floatImg;
                //floatImg.convertTo(gray, CV_32F);
                //printMat(theFrame);
                //vector<int>vec = asciiToDecimal("D:\\CUDA_WLI\\Data Acquisition\\BaslerImageFloat.csv");
               // writeDecimalToCsv("D:\\CUDA_WLI\\Data Acquisition\\BaslerImageFloat.csv", vec);

                 //Save openCV Mat frame
                char str[15];
                tCount = clock() - tStart;
                sprintf(str, "%f seconds", ((float)tCount) / CLOCKS_PER_SEC);
                printf("%ld: %f seconds\n", tCount, ((float)tCount) / CLOCKS_PER_SEC);
                //sleep(1);
                this_thread::sleep_for(chrono::milliseconds(1));


                tCount = clock() - tStart;
                sprintf(str, "%f seconds", ((float)tCount) / CLOCKS_PER_SEC);
                printf("%ld: %f seconds\n", tCount, ((float)tCount) / CLOCKS_PER_SEC);

               // putText(theFrame, str, cvPoint(30, 100), FONT_HERSHEY_SIMPLEX, 2, cvScalar(200, 200, 250), 3, CV_AA);
                putText(theFrame, str, cvPoint(30, 100), FONT_HERSHEY_SIMPLEX, 2, cvScalar(200, 200, 250), 3, cv::LINE_AA);
                imwrite("LastImg.BMP", theFrame);



                // keep a copy of it
                //cv::Mat myFrame;
                //theFrame.copyTo(myFrame); // myFrame life cycle is now under your control

                // // Convert of opencv format and display
                // CImageFormatConverter fc;
                // //fc.OutputPixelFormat = PixelType_BGR8packed;
                // fc.OutputPixelFormat = PixelType_Mono8;
                // CPylonImage image;
                // fc.Convert(image, ptrGrabResult);

                // Mat cv_img = cv::Mat(ptrGrabResult->GetHeight(), ptrGrabResult->GetWidth(), CV_8UC1,(uint8_t*)image.GetBuffer());

                // printMat(cv_img);

                // imshow("src_window", cv_img);  // display the image in OpenCV image window
                // waitKey(0);

            }
            else
            {
                cout << "Error: " << ptrGrabResult->GetErrorCode() << " " << ptrGrabResult->GetErrorDescription() << endl;
            }
        }
    }
    catch (GenICam::GenericException& e)
    {
        // Error handling.
        cerr << "An exception occurred." << endl
            << e.GetDescription() << endl;
        exitCode = 1;
    }

    // Comment the following two lines to disable waiting on exit.
    // cerr << endl << "Press Enter to exit." << endl;
    // while( cin.get() != '\n');

    return exitCode;
}