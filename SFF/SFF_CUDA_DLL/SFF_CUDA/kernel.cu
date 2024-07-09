#define BUILD_CUDA_MATH

//For CUDA
#include "CUDASFF.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//For CPP
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <string>
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

//For Thrust
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

int IMG_SIZE = 1; //default
int ZposSize = 1;
double** devicePtrs;

extern "C" EXP_CUDA_MATH int setImgCount(int imgNo)
{
    IMG_SIZE = imgNo;
    ZposSize = imgNo;
    devicePtrs = new double*[IMG_SIZE];

    return imgNo;
}

//CPU Global vectors

std::vector<cv::Mat>original_img_stack;
cv::Mat GrayImage;
std::vector<cv::Mat>cpuImgStack;


int height;
int width;
cv::Mat zHeight;



//GPU Global vectors
std::vector<double>zPos;
std::vector <cv::cuda::GpuMat>gpuImgStack;

cv::cuda::GpuMat maxIndices; //for storing max index values
std::vector < cv::cuda::GpuMat>SML3;
std::vector<cv::cuda::GpuMat> ML3;


double** d_SML3;
cv::cuda::GpuMat max_gauss;


//Functions
void convertImage();
void startGPU();
void SML();

__global__ void convolution_Kernel(double* inputImg, double* convolutedImg, int imgWidth, int imgHeight)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x; //blockIdx.x = block index,  blockDim.x = no of threads in a block,  threadIdx.x = index of thread within a block
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < 2 || col < 2 || row >= imgHeight - 3 || col >= imgWidth - 3)
        return;

    double kernel_h[3][3] = { -1.0 , 2.0 , -1.0 ,
                             0.0 , 0.0 , 0.0 ,
                               0.0 , 0.0 , 0.0 };

    double kernel_v[3][3] = { 0.0 ,-1.0 ,0.0,
                              0.0 ,2.0 ,0.0 ,
                              0.0 , -1.0 , 0.0 };

    double sumX = 0.0, sumY = 0.0, color=0.0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            color = inputImg[(row + j) * imgWidth + (col + i)];
            sumX += color * kernel_h[i + 1][j + 1];
            sumY += color * kernel_v[i + 1][j + 1];
        }
    }
    
    double sum = 0.0;
    sum = std::abs(sumX) + std::abs(sumY);
    if (sum > 255) sum = 255;
    if (sum < 0) sum = 0;

    convolutedImg[row * imgWidth + col] = sum;
}

__global__ void Sum_Mask_kernel(double* inputImg, double* convolutedImg, int imgWidth, int imgHeight)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < 4 || col < 4 || row >= imgHeight - 4 || col >= imgWidth - 4)
        return;

    double sum = 0.0, color = 0.0;
    for (int j = -4; j <= 4; j++) { //9x9 kernel of 1's
        for (int i = -4; i <= 4; i++) {
            color = inputImg[(row + j) * imgWidth + (col + i)];
            sum += color * 1.0;
        }

    }
    convolutedImg[row * imgWidth + col] = sum;
}

__global__ void MaxIndices_Kernel(double** SML3, double* maxIndices, int imgWidth, int imgHeight, int size)
{
   
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= imgHeight || col >= imgWidth)
       return;

    double maxIntensity = -1.0;
    int currentIndex = 0;
    double intensity = 0.0;
    int index;

    for (index = 0; index < size; index++) {
        //double* img = SML3[index];
        intensity = SML3[index][row * imgWidth + col];
        if (intensity > maxIntensity) {
            maxIntensity = intensity;
            currentIndex = index;
        }
    }
    maxIndices[row * imgWidth + col] = (double)(currentIndex);
   
}

__global__ void GPF_Kernel_01(double* d1, double* d3, int width, int height, int size)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int index = row * width + col;

    if (d1[index] < 0.0)
    {
        d1[index] = 0.0;
    }
    if ((int)d3[index] >= size)
    {
        d3[index] = (double)size - 1;
    }
}

__global__ void GPF_Kernel_02 (double** SML3, double* d1, double* d2, double* d3, double* f1, double* f2, double* f3, int width, int height, int size)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= height || col >= width)
        return;
    int index = row * width + col;

    int d1Val = (int)d1[index];
    int d2Val = (int)d2[index];
    int d3Val = (int)d3[index];

    if (d1Val >= size || d2Val >= size || d3Val >= size)
        return;

    /*double* d1Img = SML3[d1Val];
    double* d2Img = SML3[d2Val];
    double* d3Img = SML3[d3Val];*/

    /*f1[index] = d1Img[index];
    f2[index] = d2Img[index];
    f3[index] = d3Img[index];*/

    f1[index] = SML3[d1Val][index];
    f2[index] = SML3[d2Val][index];
    f3[index] = SML3[d3Val][index];
}

__global__ void GPF_Kernel_03(double* f1, double* f2, double* f3, double* d1, double* d2, double* d3, double* max_gauss, int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= height || col >= width)
        return;

    int index = row * width + col;

    double d_num1 = 0.0, d_num2 = 0.0, d_denom = 0.0;

    d_num1 = (double) (log(f2[index]) - log(f3[index])) * (pow(d2[index], 2) - (pow(d1[index], 2)));
    d_num2 = (double) (log(f2[index]) - log(f1[index])) * (pow(d2[index], 2) - (pow(d3[index], 2)));
    d_denom = 2 * (2 * log(f2[index]) - log(f1[index]) - log(f3[index]));

    max_gauss[index] = (d_num1/d_denom) - (d_num2/d_denom);
    if (max_gauss[index] != max_gauss[index]) //handling NAN values
    {
        max_gauss[index] = d2[index];
    }
}

//void Polyfit()
//{
//    int N = zPos.size();
//
//    std::vector<double>x(N, 0.0);
//    std::vector<double>y(N, 0.0);
//    
//    for (int i = 0; i < N; i++) {
//        x[i] = i + 1;
//    }
//
//    for (int i = 0; i < N; i++) {
//        y[i] = zPos[i];
//    }
//
//    int n = 2; //polynomial degree
//    std::vector<double> X(2 * n + 1,0.0);
//    for (int i = 0; i < 2 * n + 1; i++) {
//        X[i] = 0.0;
//        for (int j = 0; j < N; j++) {
//            X[i] = X[i] + std::pow(x[j], i);
//        }
//    }
//
//    std::vector<std::vector<double>>B(n + 1, std::vector<double>(n + 2,0.0));
//    std::vector<double>a(n + 1,0.0);
//    for (int i = 0; i <= n; i++) {
//        for (int j = 0; j <= n; j++) {
//            B[i][j] = X[i + j];
//        }
//    }
//
//    std::vector<double>Y(n + 1);
//    for (int i = 0; i < n + 1; i++) {
//        Y[i] = 0;
//        for (int j = 0; j < N; j++)
//            Y[i] = Y[i] + pow(x[j], i) * y[j];
//    }
//
//    for (int i = 0; i <= n; i++) {
//        B[i][n + 1] = Y[i];
//    }
//
//    n = n + 1;
//    for (int i = 0; i < n; i++) {
//        for (int k = i + 1; k < n; k++) {
//            if (B[i][i] < B[k][i]) {
//                for (int j = 0; j <= n; j++) {
//                    double temp = B[i][j];
//                    B[i][j] = B[k][j];
//                    B[k][j] = temp;
//                }
//            }
//        }
//    }
//
//    for (int i = 0; i < n - 1; i++) {
//        for (int k = i + 1; k < n; k++) {
//            double t = B[k][i] / B[i][i];
//            for (int j = 0; j <= n; j++) {
//                B[k][j] = B[k][j] - t * B[i][j];
//            }
//        }
//    }
//    for (int i = n - 1; i >= 0; i--) {
//        a[i] = B[i][n];
//        for (int j = 0; j < n; j++) {
//            if (j != i) {
//                a[i] = a[i] - B[i][j] * a[j];
//            }
//        }
//        a[i] = a[i] / B[i][i];
//    }
//    zHeight.release();
//    max_gauss.download(zHeight); //zHeight contains all data points for 3D geometry
//
//    for (int row = 0; row < height; row++) {
//        for (int col = 0; col < width; col++) {
//            zHeight.at<double>(row, col) = a[0] + a[1] * zHeight.at<double>(row, col) + a[2] * zHeight.at<double>(row, col) * zHeight.at<double>(row, col);
//        }
//    }
//
//}

void GPF_fast()
{
    //Create matrices to represent d1, d2, d3 for all points
    cv::cuda::GpuMat d1 = cv::cuda::GpuMat(height, width, CV_64F);
    //cv::cuda::GpuMat d2 = cv::cuda::GpuMat(height, width, CV_64F);
    cv::cuda::GpuMat d3 = cv::cuda::GpuMat(height, width, CV_64F);

    cv::cuda::add(maxIndices, cv::cuda::GpuMat(maxIndices.size(), maxIndices.type(), cv::Scalar(-1)), d1);
    cv::cuda::add(maxIndices, cv::cuda::GpuMat(maxIndices.size(), maxIndices.type(), cv::Scalar(1)), d3);
    //d2 = maxIndices;

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    GPF_Kernel_01 << <grid, block >> > (d1.ptr<double>(), d3.ptr<double>(), width, height, IMG_SIZE);
    cudaDeviceSynchronize();


    cv::cuda::GpuMat f1 = cv::cuda::GpuMat(height, width, CV_64F);
    cv::cuda::GpuMat f2 = cv::cuda::GpuMat(height, width, CV_64F);
    cv::cuda::GpuMat f3 = cv::cuda::GpuMat(height, width, CV_64F);

    GPF_Kernel_02 << <grid, block >> > (d_SML3, d1.ptr<double>(), maxIndices.ptr<double>(), d3.ptr<double>(), f1.ptr<double>(), f2.ptr<double>(), f3.ptr<double>(), width, height, IMG_SIZE);
    cudaDeviceSynchronize();
    
    max_gauss = cv::cuda::GpuMat(height, width, CV_64F);

    GPF_Kernel_03 << <grid, block >> > (f1.ptr<double>(), f2.ptr<double>(), f3.ptr<double>(), d1.ptr<double>(), maxIndices.ptr<double>(), d3.ptr<double>(), max_gauss.ptr<double>(), width, height);
    cudaDeviceSynchronize();

    //for (int i = 0; i < IMG_SIZE; i++) { //make free later 
    //    cudaFree(d_SML3[i]);
    //    free(devicePtrs[i]);
    //}
    cudaFree(d_SML3);
    delete[]devicePtrs;

    //Add
    zHeight.release();
    max_gauss.download(zHeight); //zHeight contains all data points for 3D geometry
}

void SML()
{
    height = gpuImgStack[0].rows;
    width = gpuImgStack[0].cols;

    //For horizontal 

    for (int i = 0; i < IMG_SIZE; i++) {
        ML3.push_back(cv::cuda::GpuMat(height, width, CV_64F)); //initializing as double
        SML3.push_back(cv::cuda::GpuMat(height, width, CV_64F)); //initializing as double

    }

    //Kernel Variable , can be changed depends on image size
    dim3 block(16, 16); //16*16 = 256
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y); //80*64 = 5120. So, total threads 5120*256 = 1,310,720. Thus, 1024*1280 = 1,310,720 pixels

    for (int i = 0; i < IMG_SIZE; i++) {
        convolution_Kernel << <grid, block >> > (gpuImgStack[i].ptr<double>(), ML3[i].ptr<double>(), width, height);
    }
        cudaDeviceSynchronize();


    ////Calling kernel
    for (int i = 0; i < IMG_SIZE; i++) {
        Sum_Mask_kernel << <grid, block >> > (ML3[i].ptr<double>(), SML3[i].ptr<double>(), width, height);
    }
        cudaDeviceSynchronize();
    
   for (int i = 0; i < IMG_SIZE; ++i) {
       devicePtrs[i] = SML3[i].ptr<double>();
   }

   cudaMallocManaged(&d_SML3, IMG_SIZE * sizeof(double*));
   cudaMemcpy(d_SML3, devicePtrs, IMG_SIZE * sizeof(double*), cudaMemcpyHostToDevice);

   maxIndices = cv::cuda::GpuMat(height, width, CV_64F);

   MaxIndices_Kernel <<< grid, block >>> (d_SML3, maxIndices.ptr<double>(), width, height, IMG_SIZE);
   cudaDeviceSynchronize();

   
}



void convertImage() //conversion and uploading to GPU
{
    for (int i = 0; i < IMG_SIZE; i++)
    {
        gpuImgStack[i].cv::cuda::GpuMat::convertTo(gpuImgStack[i], -1, 1, -155);
        gpuImgStack[i].cv::cuda::GpuMat::convertTo(gpuImgStack[i], -1, 3.6, 0);

        cv::cuda::cvtColor(gpuImgStack[i], gpuImgStack[i], cv::COLOR_BGR2GRAY);
        gpuImgStack[i].cv::cuda::GpuMat::convertTo(gpuImgStack[i],CV_64F);
    }
}

__global__ void gpuStartKernel(double* arr, double* summation, double b)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    summation[idx] = arr[idx] + b;
}

void startGPU()
{
    double* arr;
    double* summation;
    const int N = 2;
    double b = 100.0;

    cudaMallocManaged(&arr, N * sizeof(double));
    cudaMallocManaged(&summation, N * sizeof(double));

    for (int i = 0; i < N; ++i)
    {
        arr[i] = i + 1;
    }
    gpuStartKernel << <1, 2 >> > (arr, summation, b);
    cudaDeviceSynchronize();

    cudaFree(arr);
    cudaFree(summation);
}

extern "C" EXP_CUDA_MATH bool getMat(double* heightDataClient)
{
    try {

        //startGPU(); //Function to start GPU to decrease the overall time, 5th func
        convertImage(); //Function to read the images, 6th func
        SML();
        GPF_fast();
        //Polyfit();

        //clearing memory
        gpuImgStack.clear();
        ML3.clear();
        SML3.clear();
        maxIndices.release();
        max_gauss.release();


        int idx = 0;
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                //heightDataClient[idx] = 53.2;
                heightDataClient[idx] = zHeight.at<double>(row, col);
                idx++;
            }
        }
        return true;
    }
    catch(std::string e)
    {
        return false;
    }
    
}

extern "C" EXP_CUDA_MATH bool UploadZposDLL(double* ptr)
{
    startGPU(); //starting GPU

    zPos.clear();
    for (int i = 0; i < ZposSize; i++)
    {
        zPos.push_back(ptr[i]);
    }

    if (zPos.empty())
    {
        return false;
    }
    else
    {
        return true;
    }
}

extern "C" EXP_CUDA_MATH bool UploadImgDLL(cv::Mat img, int imgNo)
{
    cv::cuda::GpuMat gpuTempImg;
    gpuTempImg.upload(img);
  
    gpuImgStack.push_back(gpuTempImg);
    gpuTempImg.release();

    if (gpuImgStack.back().empty())
    {
        return false;
    }
    else
    {
        return true;
    }
}

