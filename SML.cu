//For CUDA
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

//Paths 
#define IMG_SIZE 162 //Change
#define IMG_READ_PATH "D:\\CUDA_WLI\\SFF\\IntensityModeMasum\\SFF_Masum\\UTAC_Data\\"
//#define IMG_READ_PATH "D:\\CUDA_WLI\\SFF\\Intensity_Mode_Test_Data_Smart_Mot_Z_Axis\\DATA_9\\"

//CPU Global vectors

cv::Mat GrayImage[IMG_SIZE];
cv::Mat cpuImgStack[IMG_SIZE];
cv::Mat original_img_stack[IMG_SIZE];
int height;
int width;


//GPU Global vectors
thrust::device_vector<double>zPos; //contains z position
cv::cuda::GpuMat gpuImgStack[IMG_SIZE];
cv::cuda::GpuMat maxIndices; //for storing max index values
cv::cuda::GpuMat SML3[IMG_SIZE];
double* devicePtrs[IMG_SIZE];
double** d_SML3;
cv::cuda::GpuMat max_gauss;


//CV_32FC1 one channel (C1) of 32-bit floating point numbers (32F). The 'C1' means one channel.

//Functions
void readZPosition(std::string csv_path);
void readImage(std::string img_path);
void releaseMemory();
void startGPU();
void SML();

void printMat(cv::Mat img)
{
    std::ofstream file;
    file.open("D:\\CUDA_WLI\\SFF\\SFF_CUDA\\Result\\maxGuassGPU.csv");

    if (!file.is_open()) {
        std::cerr << "Failed to open the file!" << std::endl;
        return;
    }
   
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            file << img.at<double>(i, j)<<",";
            //if (j != img.cols - 1) file << ", ";  // Avoid comma at the end of the line
        }
        file << "\n";  // Newline for the next row
    }

    // Close the file
    file.close();

    std::cout << "Image data written to CSV file successfully." << std::endl;
}

void gpuTocpu(cv::cuda::GpuMat& img)
{
    cv::Mat test;
    img.download(test);
    std::cout << "First Pixel: " << test.at<double>(0, 0) << "\n";
  
    printMat(test);
}

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

    if (row < 2 || col < 2 || row >= imgHeight - 3 || col >= imgWidth - 3)
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
        double* img = SML3[index];
        intensity = img[row * imgWidth + col];
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

    double* d1Img = SML3[d1Val];
    double* d2Img = SML3[d2Val];
    double* d3Img = SML3[d3Val];

    f1[index] = d1Img[index];
    f2[index] = d2Img[index];
    f3[index] = d3Img[index];
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
    //printf("%lf, %lf, %lf\n", d_num1, d_num2, d_denom);
    if (max_gauss[index] != max_gauss[index])
    {
        max_gauss[index] = d2[index];
    }
    
   // printf("%lf\n", max_gauss[index]);
}

void Polyfit()
{
    int N = zPos.size();

    thrust::device_vector<double>x(N, 0.0);
    thrust::device_vector<double>y(N,0.0);

    for (int i = 0; i < N; i++) {
        x[i] = i + 1;
    }

    for (int i = 0; i < N; i++) {
        y[i] = zPos[i];
    }

    int n = 2;
    thrust::device_vector<double>X(2 * n + 1, 0.0);
    for (int i = 0; i < 2 * n + 1; i++)
    {
        for (int j = 0; j < N; j++)
        {
            X[i] = X[i] + std::pow(x[j], i);
        }
    }

    thrust::device_vector<double>B((n + 1) * (n + 2));
    thrust::device_vector<double>a(n + 1);

    for (int i = 0; i <= n; i++) {
        for (int j = 0; j <= n+1; j++) {
           B[i * (n + 2) + j] = X[i + j]; // Can be error
        }
    }

    thrust::device_vector<double>Y(n + 1, 0.0);
    for (int i = 0; i < n + 1; i++)
    {
        for (int j = 0; j < N; j++)
        {
            Y[i] = Y[i] + std::pow(x[j], i) * y[j];
        }
    }

    for (int i = 0; i <= n; i++) {
        for (int j = 0; j <= n + 1; j++) {
            B[i * (n + 2) + (n + 1)] = Y[i]; // Can be error
        }
    }

   /* n = n + 1;

    for (int i = 0; i < n; i++)
    {
        for (int k = i + 1; k < n; k++)
        {
            if (B[i * (n + 2) + i] < B[k * (n + 2) + i])
            {
                for (int j = 0; j <= n+1; j++)
                {
                    double temp = B[i * (n + 2) + j];
                    B[i * (n + 2) + j] = B[k * (n + 2) + j];
                    B[k * (n + 2) + j] = temp;
                }
            }
        }
    }*/


    //thrust::host_vector<double> h_B = B; // Copy device vector to host vector for printing
    //for (int i = 0; i <= n; i++) {
    //    for (int j = 0; j <= n + 1; j++) {
    //        std::cout << h_B[i * (n + 2) + j] << " "; // Access the element at (i, j) in the flattened array
    //    }
    //    std::cout << std::endl;
    //}

    std::cout << "Till now ok\n";

}

void GPF_fast()
{
    //Create matrices to represent d1, d2, d3 for all points
    cv::cuda::GpuMat d1 = cv::cuda::GpuMat(height, width, CV_64F);
    cv::cuda::GpuMat d2 = cv::cuda::GpuMat(height, width, CV_64F);
    cv::cuda::GpuMat d3 = cv::cuda::GpuMat(height, width, CV_64F);

    cv::cuda::add(maxIndices, cv::cuda::GpuMat(maxIndices.size(), maxIndices.type(), cv::Scalar(-1)), d1);
    cv::cuda::add(maxIndices, cv::cuda::GpuMat(maxIndices.size(), maxIndices.type(), cv::Scalar(1)), d3);
    d2 = maxIndices;

   /* .rows = height;
    .cols = width;*/
    dim3 block(16, 16); //16*16 = 256
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    GPF_Kernel_01 << <grid, block >> > (d1.ptr<double>(), d3.ptr<double>(), width, height, IMG_SIZE);
    cudaDeviceSynchronize();


    cv::cuda::GpuMat f1 = cv::cuda::GpuMat(height, width, CV_64F);
    cv::cuda::GpuMat f2 = cv::cuda::GpuMat(height, width, CV_64F);
    cv::cuda::GpuMat f3 = cv::cuda::GpuMat(height, width, CV_64F);

    //Till OK

    GPF_Kernel_02 << <grid, block >> > (d_SML3, d1.ptr<double>(), d2.ptr<double>(), d3.ptr<double>(), f1.ptr<double>(), f2.ptr<double>(), f3.ptr<double>(), width, height, IMG_SIZE);
    cudaDeviceSynchronize();
    

    max_gauss = cv::cuda::GpuMat(height, width, CV_64F);

    GPF_Kernel_03 << <grid, block >> > (f1.ptr<double>(), f2.ptr<double>(), f3.ptr<double>(), d1.ptr<double>(), d2.ptr<double>(), d3.ptr<double>(), max_gauss.ptr<double>(), width, height);
    cudaDeviceSynchronize();

    //gpuTocpu(max_gauss);

    std::cout << "Not Correct Till now\n";
    //Handling all NAN values

    /*double* max_guass_ptr = max_gauss.ptr<double>();
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            max_guass_ptr[i][j]
        }
    }*/
   

    //for (int i = 0; i < IMG_SIZE; i++) { //make free later 
    //    cudaFree(d_SML3[i]);
    //}
    //cudaFree(d_SML3);
    //cudaFreeHost(devicePtrs);

}

void SML()
{
    height = cpuImgStack[0].rows;
    width = cpuImgStack[0].cols;

    //For horizontal 
    cv::cuda::GpuMat ML3[IMG_SIZE];

    for (int i = 0; i < IMG_SIZE; i++) {
        ML3[i] = cv::cuda::GpuMat(height, width, CV_64F); //initializing as double
    }

    //Kernel Variable , can be changed depends on image size
    dim3 block(16, 16); //16*16 = 256
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y); //80*64 = 5120. So, total threads 1,310,720. Thus, 1024*1280 = 1,310,720 pixels

    for (int i = 0; i < IMG_SIZE; i++) {
        convolution_Kernel << <grid, block >> > (gpuImgStack[i].ptr<double>(), ML3[i].ptr<double>(), width, height);
        cudaDeviceSynchronize();
    }

    for (int i = 0; i < IMG_SIZE; i++) {
        SML3[i] = cv::cuda::GpuMat(height, width, CV_64F); //initializing as double
    }

    ////Calling kernel
    for (int i = 0; i < IMG_SIZE; i++) {
        Sum_Mask_kernel << <grid, block >> > (ML3[i].ptr<double>(), SML3[i].ptr<double>(), width, height);
        cudaDeviceSynchronize();
    }
    
   // ML3->release();

   for (int i = 0; i < IMG_SIZE; ++i) {
       devicePtrs[i] = SML3[i].ptr<double>();
   }

   cudaMallocManaged(&d_SML3, IMG_SIZE * sizeof(double*));
   cudaMemcpy(d_SML3, devicePtrs, IMG_SIZE * sizeof(double*), cudaMemcpyHostToDevice);

   maxIndices = cv::cuda::GpuMat(height, width, CV_64F);

   MaxIndices_Kernel <<< grid, block >>> (d_SML3, maxIndices.ptr<double>(), width, height, IMG_SIZE);
   cudaDeviceSynchronize();


   /* clock_t cpu_start, cpu_end;
   cpu_start = clock();*/
   //cpu_end = clock();
  
   /*printf("Measuremnt Time : %4.6f \n",
      (double)((double)(cpu_end - cpu_start) / CLOCKS_PER_SEC));*/
   std::cout << "SML Complete\n" << std::endl;
}

void readImage(std::string img_path)
{
    for (int i = 0; i < IMG_SIZE; i++)
    {
        original_img_stack[i] = cv::imread(img_path + "a1_" + std::to_string(i + 1) + ".BMP");
        if (original_img_stack[i].empty())
        {
            printf("Image read failed\n");
            exit(-1);
        }
       // std::cout << i <<" IMG = " << i + 1 << std::endl;
    }
    
    std::cout << "Image Loading Done!" << std::endl;
    for (int i = 0; i < IMG_SIZE; i++)
    {
        cv::cvtColor(original_img_stack[i], GrayImage[i], cv::COLOR_BGR2GRAY);
        GrayImage[i].convertTo(cpuImgStack[i], CV_64F);

        gpuImgStack[i].upload(cpuImgStack[i]);
        if (gpuImgStack[i].empty())
        {
            std::cout << "Not uploaded\n";
        }
    }
    //printMat(cpuImgStack[0]);
//    gpuTocpu(gpuImgStack[0]);
}

void readZPosition(std::string csv_path)
{
    std::string str = csv_path + "a1.csv";
    std::ifstream file(str);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Failed to open the file." << std::endl;
        return;
    }
    //skip the first row
    getline(file, line);

    // Read the file line by line
    while (getline(file, line)) {
        std::istringstream sstream(line);
        std::string cell;
        int columnCount = 0;

        // Extract each cell in the row
        while (getline(sstream, cell, ',')) {
            columnCount++;
            if (columnCount == 2) {  // Check if it's the second column
                zPos.push_back(stod(cell));  // Add the second column cell to the vector
                break;  // No need to continue to the end of the line
            }
        }
    }
    file.close();  
}

__global__ void gpuStartKernel(double* arr, double* summation, double b)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    summation[idx] = arr[idx] + b;
   // printf("%lf\n", summation[idx]);
}

void startGPU()
{
    //Add 100 to all the elements of the array
    double* arr;
    double* summation;
    const int N = 10;
    double b = 100.0;

    cudaMallocManaged(&arr, N * sizeof(double));
    cudaMallocManaged(&summation, N * sizeof(double));

    for (int i = 0; i < N; ++i)
    {
        arr[i] = i + 1;
    }
    gpuStartKernel << <1, 10 >> > (arr, summation, b);
    cudaDeviceSynchronize();

    cudaFree(arr);
    cudaFree(summation);
}

int main() //look at memory alloc and dealloc at the end
{
    std::cout << "Program Starts\n";
    startGPU(); //Function to start GPU to decrease the overall time
    readZPosition(IMG_READ_PATH); //Function to read the z pos vals
    readImage(IMG_READ_PATH); //Function to read the images
    SML();
    GPF_fast();
    Polyfit();
    std::cout << "Till now OK\n";

   
   
    //releaseMemory();

    std::getchar();

    return 0;
}

