#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include"PSOHeader.h"

//For CPP
#include <iostream>
#include <stdio.h> 
#include <fstream>
#include <sstream>

//For openCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp> //for filtering
#include <opencv2/cudafilters.hpp>  //for filtering
#include <opencv2/cudaarithm.hpp> //for abs
#include <opencv2/imgcodecs.hpp>     // Image file reading and writing

//For pointer in GPU
#include<thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>

//Paths
#define IMG_SIZE 415 //Change
#define IMG_WRITE_PATH "E:\\Interferometry\\RAFT\\bump_img_wli_raft\\"
#define IMG_READ_PATH "E:\\Interferometry\\RAFT\\bump_img_wli_raft\\"

using namespace std;
using namespace cv;

std::vector<double>zPos;
std::vector < cv::cuda::GpuMat>gpuImgStack;
cv::Mat original_img;
std::vector<cv::Mat>GrayImage;
std::vector<cv::Mat>cpuImgStack;

//Background image
cv::Mat tempImBG_;
cv::cuda::GpuMat ImBG_;


//Note: 
//Start from minMax function 

//Read Z position
void readZPosition(std::string csv_path)
{

    SAmp sAmp;
    string path = csv_path + "bump_img_wli_raft.CSV";

    std::ifstream file(path);  // Open the CSV file
    if (!file.is_open()) {
        //std::cerr << "Error opening file!" << std::endl;
        return;
    }
    int cnt = 0;

    std::vector<double> columnData;  // Vector to store 2nd column values
    std::string line;

     //Read the CSV file line by line
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        int colIndex = 0;

        // Read each cell in the line
        while (std::getline(ss, cell, ',')) {
            if (colIndex == 1) {  // 2nd column (0-based index)
                try {
                    columnData.push_back(std::stod(cell));  // Convert to double and store
                    sAmp.PzPos_um = columnData[cnt];
                    ImgInfo.push_back(sAmp);
                    cnt++;
                }
                catch (const std::exception& e) {
                    std::cerr << "Invalid data: " << cell << std::endl;
                }
            }
            colIndex++;
        }
    }

    file.close();
}

//Read images
void readImage(std::string img_path) //conversion and uploading to GPU
{
    //Clearing memory
    gpuImgStack.clear();

    cv::cuda::GpuMat gpuTempImg;
    for (int i = 0; i < IMG_SIZE; i++)
    {
        original_img = cv::imread(img_path + "bump_img_wli_raft_" + std::to_string(i + 1) + ".BMP");
        if (original_img.empty())
        {
            printf("Image read failed\n");
            exit(-1);
        }
        gpuTempImg.upload(original_img);
        gpuImgStack.push_back(gpuTempImg);

       // gpuImgStack[i].cv::cuda::GpuMat::convertTo(gpuImgStack[i], -1, 1, -155);
      //  gpuImgStack[i].cv::cuda::GpuMat::convertTo(gpuImgStack[i], -1, 3.6, 0);

       // cv::cuda::cvtColor(gpuImgStack[i], gpuImgStack[i], cv::COLOR_BGR2GRAY);
       // gpuImgStack[i].cv::cuda::GpuMat::convertTo(gpuImgStack[i], CV_64F);

    }

    std::cout << "Image UpLoading Done!" << std::endl;

}

//Init Calculation

void GetDim(int& x, int& y, int& b) {
    x = gpuImgStack[0].cols; y = gpuImgStack[0].rows; b = 24;
}

bool InitCalc()
{
    SFrng F;

    nSteps = int(gpuImgStack.size());
    if (nSteps < 1) return false;
    zRange_um = ImgInfo[nSteps - 1].PzPos_um - ImgInfo[0].PzPos_um;
    UStep_um = zRange_um / nSteps;

    tempImBG_ = cv::imread("E:\\Interferometry\\RAFT\\bump_img_wli_raft\\eeee.BMP");
    ImBG_.upload(tempImBG_);

    GetDim(wd, ht, bpp);

    int Ch1 = REDA, Ch2 = GRNA, Ch3 = BLUA, Ch4 = WHTA;

    wlen_um[Ch1] = 0; wlen_um[Ch2] = 0;
    wlen_um[Ch3] = 0; wlen_um[Ch4] = 0;

    int n1 = 0, n2 = 0, n3 = 0, n4 = 0;
    int dh = ht / 7, dw = wd / 7;
    SROI R;
    R.i2 = nSteps;



}

float* Get(FRP Ch, int ist, int sz)
{
    if (sz < 1) {
        return nullptr;
    }
    // Pointer to store result
    float* result_ptr = nullptr;

    switch (Ch)
    {
    case ZAXS:
        if (Sz[Ch] != sz) {
            Sz[Ch] = sz;
            zaxs_d.resize(sz);
        }
        // Get raw pointer to device memory
        result_ptr = thrust::raw_pointer_cast(zaxs_d.data() + ist);
        break;

    case REDA:
        if (Sz[Ch] != sz) {
            Sz[Ch] = sz;
            reda_d.resize(sz);
        }
        result_ptr = thrust::raw_pointer_cast(reda_d.data() + ist);
        break;

    case GRNA:
        if (Sz[Ch] != sz) {
            Sz[Ch] = sz;
            grna_d.resize(sz);
        }
        result_ptr = thrust::raw_pointer_cast(grna_d.data() + ist);
        break;

    case BLUA:
        if (Sz[Ch] != sz) {
            Sz[Ch] = sz;
            blua_d.resize(sz);
        }
        result_ptr = thrust::raw_pointer_cast(blua_d.data() + ist);
        break;

    case WHTA:
        if (Sz[Ch] != sz) {
            Sz[Ch] = sz;
            whta_d.resize(sz);
        }
        result_ptr = thrust::raw_pointer_cast(whta_d.data() + ist);
        break;

    case TMP1:
        if (Sz[Ch] != sz) {
            Sz[Ch] = sz;
            tmp1_d.resize(sz);
        }
        result_ptr = thrust::raw_pointer_cast(tmp1_d.data() + ist);
        break;
    default:
        break;
    }
}

int main()
{
	std::cout << "Start";

    readZPosition(IMG_READ_PATH);
    readImage(IMG_READ_PATH);
    InitCalc();


	return 0;
}