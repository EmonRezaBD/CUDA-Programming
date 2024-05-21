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
#define IMG_SIZE 10 //Change
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
    file.open("D:\\CUDA_WLI\\SFF\\SFF_CUDA\\Result\\GPUImage8.csv");

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

void testCode()
{
    std::string str = "D:\\CUDA_WLI\\SFF\\Data\\a1\\";
    cv::Mat demo = cv::imread(str + "a1_75.BMP");
    cv::Mat gray;
    cv::cvtColor(demo, gray, cv::COLOR_BGR2GRAY);
    cv::Mat bit32;
    gray.convertTo(bit32, CV_32FC1);

    printMat(bit32);

    cv::cuda::GpuMat gpuImg;
    gpuImg.upload(bit32);

    cv::Mat dImg;
    gpuImg.download(dImg);
    printMat(dImg);
}

void gpuTocpu(cv::cuda::GpuMat& img)
{
    cv::Mat test;
    img.download(test);
    std::cout << "First Pixel: " << test.at<double>(0, 0) << "\n";
  
    printMat(test);
}

__global__ void convolution_Vertical_Kernel(double** inputImg, double** convolutedImg, int imgWidth, int imgHeight, int imgIndex)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < 2 || col < 2 || row >= imgHeight - 3 || col >= imgWidth - 3)
        return;

    double kernel_v[3][3] = { 0 ,-1 ,0,
                    0 ,-2 ,0 ,
                    0 , -1 , 0 };

    double sum = 0;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            double color = inputImg[imgIndex][(row + j) * imgWidth + (col + i)];
            sum += color * kernel_v[j + 1][i + 1];
        }
    }
    convolutedImg[imgIndex][row * imgWidth + col] = sum;
}

//__global__ void convolution_Horizonatal_Kernel(double** inputImg, double** convolutedImg, int imgWidth, int imgHeight, int imgIndex)
//{
//    int col = blockIdx.x * blockDim.x + threadIdx.x;
//    int row = blockIdx.y * blockDim.y + threadIdx.y;
//
//    if (row < 2 || col < 2 || row >= imgHeight - 3 || col >= imgWidth - 3)
//        return;
//
//    double kernel_h[3][3] = { -1 , 2 , -1 ,
//                             0 , 0 , 0 ,
//                               0 , 0 , 0 };
//
//    double sum = 0;
//    for (int j = -1; j <= 1; j++) {
//        for (int i = -1; i <= 1; i++) {
//            double color = inputImg[imgIndex][(row + j) * imgWidth + (col + i)];
//            sum += color * kernel_h[j + 1][i + 1];
//        }
//    }
//    convolutedImg[imgIndex][row * imgWidth + col] = sum;
//}

__global__ void convolution_Horizonatal_Kernel(double* inputImg, double* convolutedImg, int imgWidth, int imgHeight)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < 2 || col < 2 || row >= imgHeight - 3 || col >= imgWidth - 3)
        return;

    double kernel_h[3][3] = { -1 , 2 , -1 ,
                             0 , 0 , 0 ,
                               0 , 0 , 0 };

    double sum = 0;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            double color = inputImg[(row + j) * imgWidth + (col + i)];
            sum += color * kernel_h[j + 1][i + 1];
        }
    }
    convolutedImg[row * imgWidth + col] = sum;
}

void SML()
{
    height = cpuImgStack[0].rows;
    width = cpuImgStack[0].cols;

    //Single Pointer Start
    //cv::cuda::GpuMat* horizontalFilteredImg = new cv::cuda::GpuMat[IMG_SIZE]; //uchar
    cv::cuda::GpuMat horizontalFilteredImg[IMG_SIZE];

    for (int i = 0; i < IMG_SIZE; i++) {
        horizontalFilteredImg[i] = cv::cuda::GpuMat(height, width, CV_64F); //initializing as double
    }

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // //Calling kernel
    for (int i = 0; i < IMG_SIZE; i++) {
        convolution_Horizonatal_Kernel << <grid, block >> > (gpuImgStack[i].ptr<double>(), horizontalFilteredImg[i].ptr<double>(), width, height);
        cudaDeviceSynchronize();
    }
    
    cv::Mat hostResult;
    horizontalFilteredImg[8].download(hostResult);

    printMat(hostResult);


   // //Horizontal Kernel Start Working Code Double Pointer
   // cv::cuda::GpuMat* horizontalFilteredImg = new cv::cuda::GpuMat[IMG_SIZE]; //uchar
   // for (int i = 0; i < IMG_SIZE; i++) {
   //     horizontalFilteredImg[i] = cv::cuda::GpuMat(height, width, CV_64F); //initializing as double
   // }

   // double** d_gpuImgStack; //to pass images in gpu
   // double** d_horizontalFilteredImg; //to store convolution result

   // // Host pointers for the image stacks
   // double** h_gpuImgStack = new double* [IMG_SIZE]; //helper 
   // double** h_horizontalFilteredImg = new double* [IMG_SIZE]; //helper

   // // Allocate memory for the device pointers
   // cudaMalloc(&d_gpuImgStack, IMG_SIZE * sizeof(double*));
   // cudaMalloc(&d_horizontalFilteredImg, IMG_SIZE * sizeof(double*));

   // // Allocate memory for each image on the device and copy data
   // for (int i = 0; i < IMG_SIZE; i++) {
   //     // Allocate memory for each image on the device
   //     cudaMalloc(&h_gpuImgStack[i], height * width * sizeof(double));
   //     cudaMalloc(&h_horizontalFilteredImg[i], height * width * sizeof(double));

   //     // Copy data from GpuMat to the allocated memory
   //     cudaMemcpy(h_gpuImgStack[i], gpuImgStack[i].ptr<double>(), height * width * sizeof(double), cudaMemcpyDeviceToDevice);
   //     cudaMemcpy(h_horizontalFilteredImg[i], horizontalFilteredImg[i].ptr<double>(), height * width * sizeof(double), cudaMemcpyDeviceToDevice);
   // }

   // // Copy the array of pointers to the device
   // cudaMemcpy(d_gpuImgStack, h_gpuImgStack, IMG_SIZE * sizeof(double*), cudaMemcpyHostToDevice);
   // cudaMemcpy(d_horizontalFilteredImg, h_horizontalFilteredImg, IMG_SIZE * sizeof(double*), cudaMemcpyHostToDevice);

   // // Define block and grid sizes
   // dim3 block(16, 16);
   // dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y); 

   // //Calling kernel
   // for (int i = 0; i < IMG_SIZE; i++) {
   //     convolution_Horizonatal_Kernel << <grid, block >> > (d_gpuImgStack, d_horizontalFilteredImg, width, height, i);
   //     cudaDeviceSynchronize();
   // }

   // // Host pointers to store the result
   // double** h_resultImgs = new double* [IMG_SIZE];
   // for (int i = 0; i < IMG_SIZE; i++) 
   // {
   //     h_resultImgs[i] = new double[height * width];
   // }

   // // Copy the result back to the host
   // for (int i = 0; i < IMG_SIZE; i++) {
   //     cudaMemcpy(h_resultImgs[i], h_horizontalFilteredImg[i], height * width * sizeof(double), cudaMemcpyDeviceToHost);
   //     //cudaMemcpy(h_resultImgs[i], d_horizontalFilteredImg[i], height * width * sizeof(double), cudaMemcpyDeviceToHost);
   // }

   // // Clean up device memory
   // for (int i = 0; i < IMG_SIZE; i++) {
   //     cudaFree(h_gpuImgStack[i]);
   //     cudaFree(h_horizontalFilteredImg[i]);
   // }
   // cudaFree(d_gpuImgStack);
   // cudaFree(d_horizontalFilteredImg);

   // // Process the results or store them
   // //for (int i = 0; i < IMG_SIZE; i++) {
   //     // Convert the result to cv::Mat if needed
   //     cv::Mat resultMat(height, width, CV_64F, h_resultImgs[9]);
   //     // Process or store resultMat
   //// }

   // printMat(resultMat);

   // // Clean up host memory for results
   // for (int i = 0; i < IMG_SIZE; i++) {
   //     delete[] h_resultImgs[i];
   // }
   // delete[] h_resultImgs;
   // delete[] h_gpuImgStack;
   // delete[] h_horizontalFilteredImg;
    //Horizontal Kernel End Working Code Double Pointer


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
        // std::cout << i <<" IMG = " << i + 75 << std::endl;
    }
    
    //std::cout << "Image Loading Done!" << std::endl;
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
    //gpuTocpu(gpuImgStack[0]);
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





__global__ void findMaxIntensityIndexKernel(float** images, float* maxIndices, int width, int height, int imgSize, int step) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
     
    if (x >= width || y >= height) return;

    float maxIntensity = -FLT_MAX;
    int maxIndex = 0;

    for (int i = 0; i < imgSize; ++i) {
        //float intensity = ((float*)((char*)images[i] + y * step))[x];
        float intensity = ((float*)((float*)images[i] + y * step))[x];
       // printf("pixel: %f", intensity);

        if (intensity > maxIntensity) {
            maxIntensity = intensity;
            maxIndex = i;
        } 
    }

    maxIndices[y * width + x] = maxIndex;
   // printf("%f\n", maxIndices[y * width + x]);
}

//__global__ void conv_img_gpu_stack(cv::cuda::GpuMat imgGPUStack[], cv::cuda::GpuMat kernel, cv::cuda::GpuMat imgfGPUStack[], int Nx, int Ny, int N_images, int kernel_size)
__global__ void conv_img_gpu_stack(cv::cuda::PtrStepSz<float> imgGPUStack[], cv::cuda::PtrStepSz<float> kernel, cv::cuda::PtrStepSz<float> imgfGPUStack[], int Nx, int Ny, int N_images, int kernel_size)
{
    //printf("%f",imgfGPUStack[0].cols);

    int tid = threadIdx.x;
    int iy = blockIdx.x + (kernel_size - 1) / 2;
    int ix = threadIdx.x + (kernel_size - 1) / 2;
    int imageIdx = blockIdx.y;
    int idx = imageIdx * Nx * Ny + iy * Nx + ix;
    int K2 = kernel_size * kernel_size;
    int center = (kernel_size - 1) / 2;

    int ii, jj;
    float sum = 0.0;

    extern __shared__ float sdata[];
    if (tid < K2)
        sdata[tid] = kernel.data[tid];

    //__syncthreads(); //No error, just a warning

    if (idx < Nx * Ny)
    {
        for (int ki = 0; ki < kernel_size; ki++)
        {
            for (int kj = 0; kj < kernel_size; kj++)
            {
                ii = kj + ix - center;
                jj = ki + iy - center;

                int pixelIdx = jj * imgGPUStack[imageIdx].step + ii;
                float* pixelPtr = imgGPUStack[imageIdx].data + pixelIdx;
                sum += *pixelPtr * sdata[ki * kernel_size + kj];

            }
        }
        int outputPixelIdx = iy * imgfGPUStack[imageIdx].step + ix;
        float* outputPixelPtr = imgfGPUStack[imageIdx].data + outputPixelIdx;
        *outputPixelPtr = sum;
    }

}

