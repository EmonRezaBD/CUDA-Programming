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


//Paths 
#define IMG_SIZE 162 //Change
#define IMG_READ_PATH "D:\\CUDA_WLI\\SFF\\IntensityModeMasum\\SFF_Masum\\UTAC_Data\\"

//CPU Global vectors

cv::Mat GrayImage[IMG_SIZE];
cv::Mat cpuImgStack[IMG_SIZE];
cv::Mat max_gauss;
std::vector<double>z; //contains motor positions
cv::Mat original_img_stack[IMG_SIZE];
int height;
int width;

//GPU Global vectors
cv::cuda::GpuMat zPos(1, z.size(), CV_64F); //contains motor positions
cv::cuda::GpuMat gpuImgStack[IMG_SIZE];
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
    file.open("D:\\CUDA_WLI\\SFF\\SFF_CUDA\\Result\\GPUImg11.csv");

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

//void testCode()
//{
//    std::string str = "D:\\CUDA_WLI\\SFF\\Data\\a1\\";
//    cv::Mat demo = cv::imread(str + "a1_75.BMP");
//    cv::Mat gray;
//    cv::cvtColor(demo, gray, cv::COLOR_BGR2GRAY);
//    cv::Mat bit32;
//    gray.convertTo(bit32, CV_32FC1);
//
//    printMat(bit32);
//
//    cv::cuda::GpuMat gpuImg;
//    gpuImg.upload(bit32);
//
//    cv::Mat dImg;
//    gpuImg.download(dImg);
//    printMat(dImg);
//}

void gpuTocpu(cv::cuda::GpuMat& img)
{
    cv::Mat test;
    img.download(test);
    std::cout << "First Pixel: " << test.at<double>(0, 0) << "\n";
  
    printMat(test);
}

__global__ void convolution_Kernel(double* inputImg, double* convolutedImg, int imgWidth, int imgHeight)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
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

__global__ void findMaxIndices(double** SML3, double* maxIndices, int imgWidth, int imgHeight, int size)
{
    printf("In Kernel\n");
    double* dd = SML3[0];
    printf("First val: %lf\n ", dd[0]);

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

 

    if (row >= imgHeight || col >= imgWidth)
        return;

    double maxIntensity = -1.0;
    int currentIndex = 0;
    double intensity = 0.0;
    for (int index = 0; index < size; index++) {
        intensity = SML3[index][row * imgWidth + col];
       // intensity = SML3[row * imgWidth + col];
        //printf("%lf\n ", intensity);
        if (intensity > maxIntensity) {
            maxIntensity = intensity;
            currentIndex = index;
        }
    }

    maxIndices[row * imgWidth + col] = (double)(currentIndex);
   
}


void SML()
{
    height = cpuImgStack[0].rows;
    width = cpuImgStack[0].cols;

    clock_t cpu_start, cpu_end;
    cpu_start = clock();

    //For horizontal 
    cv::cuda::GpuMat ML3[IMG_SIZE];

    for (int i = 0; i < IMG_SIZE; i++) {
        ML3[i] = cv::cuda::GpuMat(height, width, CV_64F); //initializing as double
    }

    dim3 block(16, 16); //16*16 = 256
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y); //80*64 = 5120. So, total threads 1,310,720. Thus, 1024*1280 = 1,310,720 pixels

    //Calling kernel
    for (int i = 0; i < IMG_SIZE; i++) {
        convolution_Kernel << <grid, block >> > (gpuImgStack[i].ptr<double>(), ML3[i].ptr<double>(), width, height);
        cudaDeviceSynchronize();
    }

    cv::cuda::GpuMat SML3[IMG_SIZE];

    for (int i = 0; i < IMG_SIZE; i++) {
        SML3[i] = cv::cuda::GpuMat(height, width, CV_64F); //initializing as double
    }

    ////Calling kernel
    for (int i = 0; i < IMG_SIZE; i++) {
        Sum_Mask_kernel << <grid, block >> > (ML3[i].ptr<double>(), SML3[i].ptr<double>(), width, height);
        cudaDeviceSynchronize();
    }
    
   // ML3->release();


   /*cpu_end = clock();
   printf("Full Convo time : %4.6f \n",
       (double)((double)(cpu_end - cpu_start) / CLOCKS_PER_SEC));*/

   double** d_SML3;
   cudaMalloc(&d_SML3, IMG_SIZE * sizeof(double*));

   for (int i = 0; i < IMG_SIZE; i++) {
       cudaMemcpy(&d_SML3[i], SML3[i].ptr<double>(), sizeof(double*), cudaMemcpyHostToDevice);
   }

   double* ptr = d_SML3[0];
   printf("Ptr: %lf\n",ptr[0]);

   cv::cuda::GpuMat maxIndices; //for storing max index values
   maxIndices = cv::cuda::GpuMat(height, width, CV_64F);

   findMaxIndices <<< grid, block >>> (d_SML3, maxIndices.ptr<double>(), width, height, IMG_SIZE);
   cudaDeviceSynchronize();

   gpuTocpu(maxIndices);
   
   //double* maxIndices = new double[height * width];
   // Copy the results from the device to the host
  // cudaMemcpy(maxIndices, d_maxIndices, height * width * sizeof(double), cudaMemcpyDeviceToHost);


   // Free the device memory
  // cudaFree(d_SML3);
   //cudaFree(d_maxIndices);
   //delete[] maxIndices;

   // cv::Mat hostResult;
   // horizontalFilteredImg[161].download(hostResult);

   // printMat(hostResult);

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
                z.push_back(stod(cell));  // Add the second column cell to the vector
                break;  // No need to continue to the end of the line
            }
        }
    }
    file.close();  

    zPos.upload(cv::Mat(1, z.size(), CV_64F, z.data()));
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
    std::cout << "Till now OK\n";

   
   
    //releaseMemory();

    std::getchar();

    return 0;
}

////__global__ void conv_img_gpu_stack(cv::cuda::GpuMat imgGPUStack[], cv::cuda::GpuMat kernel, cv::cuda::GpuMat imgfGPUStack[], int Nx, int Ny, int N_images, int kernel_size)
//__global__ void conv_img_gpu_stack(cv::cuda::PtrStepSz<float> imgGPUStack[], cv::cuda::PtrStepSz<float> kernel, cv::cuda::PtrStepSz<float> imgfGPUStack[], int Nx, int Ny, int N_images, int kernel_size)
//{
//    //printf("%f",imgfGPUStack[0].cols);
//
//    int tid = threadIdx.x;
//    int iy = blockIdx.x + (kernel_size - 1) / 2;
//    int ix = threadIdx.x + (kernel_size - 1) / 2;
//    int imageIdx = blockIdx.y;
//    int idx = imageIdx * Nx * Ny + iy * Nx + ix;
//    int K2 = kernel_size * kernel_size;
//    int center = (kernel_size - 1) / 2;
//
//    int ii, jj;
//    float sum = 0.0;
//
//    extern __shared__ float sdata[];
//    if (tid < K2)
//        sdata[tid] = kernel.data[tid];
//
//    //__syncthreads(); //No error, just a warning
//
//    if (idx < Nx * Ny)
//    {
//        for (int ki = 0; ki < kernel_size; ki++)
//        {
//            for (int kj = 0; kj < kernel_size; kj++)
//            {
//                ii = kj + ix - center;
//                jj = ki + iy - center;
//
//                int pixelIdx = jj * imgGPUStack[imageIdx].step + ii;
//                float* pixelPtr = imgGPUStack[imageIdx].data + pixelIdx;
//                sum += *pixelPtr * sdata[ki * kernel_size + kj];
//
//            }
//        }
//        int outputPixelIdx = iy * imgfGPUStack[imageIdx].step + ix;
//        float* outputPixelPtr = imgfGPUStack[imageIdx].data + outputPixelIdx;
//        *outputPixelPtr = sum;
//    }
//
//}

