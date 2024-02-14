// FinalCudaOpenCV.cpp : This file contains the 'main' function. Program execution begins and ends there.

#include <iostream>
#include "opencv2/dnn.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/cuda.hpp"
#include"opencv2/core/mat.hpp"
#include "opencv2/cudaarithm.hpp"


#include <fstream>
#include<sstream>
#include<vector>
#include<time.h>

using namespace std;
using namespace cv;
using namespace dnn;
using namespace cuda;

vector<cv::Mat> cicularShift_CPU(const vector<cv::Mat>&images, int jump_point)
{
    if (images.empty()) return vector<cv::Mat>(); //returning an empty array

    vector<cv::Mat> shifted_images = images; // Copy the original images
    int num_images = images.size();

    jump_point = (jump_point % num_images + num_images) % num_images; // Normalize jump_point

    rotate(shifted_images.begin(), shifted_images.begin() + jump_point, shifted_images.end());
    return shifted_images;
}

vector<cv::cuda::GpuMat> cicularShift_GPU(vector<cv::cuda::GpuMat>& gpuImages, int jump_point)
{
    int num_images = gpuImages.size();
    if (num_images == 0) return vector<cv::cuda::GpuMat>();

    vector<cv::cuda::GpuMat> shifted_images(num_images);
    jump_point = (jump_point % num_images + num_images) % num_images; // Normalize jump_point

    for (int i = 0; i < num_images; ++i)
    {
        int new_position = (i + jump_point) % num_images;
        shifted_images[new_position] = gpuImages[i];
    }
    //gpuImages.swap(shifted_images);
    return shifted_images;

}

double checkPixels(cv::Mat& img, int x, int y)
{
    return img.at<double>(x, y);
}

void GPUProcess(vector<cv::Mat>& I3)
{
    //Remember that*** cv::gpu == cv::cuda
    vector<cv::cuda::GpuMat> gpuImages;

    clock_t start, end;
    start = clock();

    for (size_t i = 0; i < I3.size(); ++i)
    {
        cv::cuda::GpuMat OnegpuImage;
        OnegpuImage.upload(I3[i]);
        gpuImages.push_back(OnegpuImage);
    }

    end = clock();
    printf("Upload time from host to device : %4.6f sec\n",
        (double)((double)(end - start) / CLOCKS_PER_SEC));

    cout << "GPU images size: "<<gpuImages.size()<<"\n";

    //checking the pixel to verify if upload was successful
    //cv::Mat hostImg;
    //gpuImages[0].download(hostImg);
    //cout << "GPU First image's (0,0) pixel:: " << checkPixels(hostImg,0,0) <<"\n"; //118

    clock_t circular_shift_start_gpu, circular_shift_end_gpu;
    circular_shift_start_gpu = clock();

    int jump_point = 4;
    std::vector<cv::cuda::GpuMat> I2_GPU = cicularShift_GPU(gpuImages, 1 * jump_point);
    std::vector<cv::cuda::GpuMat> I1_GPU = cicularShift_GPU(gpuImages, 2 * jump_point);
    std::vector<cv::cuda::GpuMat> I4_GPU = cicularShift_GPU(gpuImages, 2 * jump_point);
    std::vector<cv::cuda::GpuMat> I5_GPU = cicularShift_GPU(gpuImages, 2 * jump_point);
    
    circular_shift_end_gpu = clock();
    printf("Circular Shifting time in GPU : %4.6f sec\n",
        (double)((double)(circular_shift_end_gpu - circular_shift_start_gpu) / CLOCKS_PER_SEC));


    I2_GPU.clear();
    I1_GPU.clear();
    I4_GPU.clear();
    I5_GPU.clear();
}

int main()
{
    cout << "Program start\n";


    clock_t cpu_start, cpu_end;
    cpu_start = clock();
 
    string path = "D:\\CUDA_WLI\\TaskHQ\\Tasks\\Task03_Feb09\\rough_cali_x10\\";
    string csvPath = path + "rough_cali_x10.CSV";

    std::ifstream file(csvPath);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << csvPath << std::endl;
        return 1;
    }

    string line;
    vector<double>position;

    //skip the first line 
    getline(file, line);
    //Read data from second line
    while (getline(file, line))
    {
        istringstream sstream(line);//full row
        string feild;

        //skip the first column
        getline(sstream,feild,',');
        //Read the remianing columns

        while (getline(sstream, feild, ','))
        {
            position.push_back(stod(feild));
        }
        //position.push_back(row);
    }

    int num_image = position.size(); //277
    double step = (position[position.size() - 1] - position[0]) / num_image; //0.0180
    vector<double> z(position.begin()+9, position.end()-9);

   // string firstImgPath = path + "rough_cali_x10_1.bmp";
    //cv::Mat t_im = cv::imread(firstImgPath);

    vector<cv::Mat>I3;

    for (int j = 10; j <= num_image - 8; ++j)
    //for (int j = 10; j <= 11; ++j)
    {
        string filename = path + "rough_cali_x10_" + std::to_string(j) + ".bmp";
        // Load the current image in grayscale directly
        cv::Mat curr_img = cv::imread(filename, cv::IMREAD_GRAYSCALE);

        if (curr_img.empty())
        {
            std::cerr << "Failed to load image: " << filename << "\n";
        }
        // Convert the image to double precision (if necessary)
        cv::Mat curr_img_double;
        curr_img.convertTo(curr_img_double, CV_64F);
        I3.push_back(curr_img_double);
    }
    cout<<"image load done"<<"\n";

    cpu_end = clock();
    printf("CPU other inst time before Uploading image to GPU : %4.6f sec\n",
        (double)((double)(cpu_end - cpu_start) / CLOCKS_PER_SEC));

    GPUProcess(I3);

    //check data of I3  
    int rowI3 = I3[0].rows; // Number of rows in the image
    int colI3 = I3[0].cols; // Number of columns in the image
    cout << "CPU First image's (0,0) pixel: " << checkPixels(I3[0], 0, 0) << "\n"; //118
    cout << "CPU First image's (row-1,col-1)/last pixel: " << checkPixels(I3[0], rowI3 - 1, colI3 - 1) << "\n"; //106
    cout << "CPU I3's Last image's (row-1,col-1)/last pixel: "<< checkPixels(I3[I3.size()-1], rowI3-1, colI3-1)<<"\n";//101

    clock_t circular_shift_start_cpu, circular_shift_end_cpu;
    circular_shift_start_cpu = clock();

    int jump_point = 4;
    std::vector<cv::Mat> I2 = cicularShift_CPU(I3, 1 * jump_point);
    std::vector<cv::Mat> I1 = cicularShift_CPU(I3, 2 * jump_point);
    std::vector<cv::Mat> I4 = cicularShift_CPU(I3, 2 * jump_point);
    std::vector<cv::Mat> I5 = cicularShift_CPU(I3, 2 * jump_point);

    circular_shift_end_cpu = clock();
    printf("Circular Shifting time in CPU : %4.6f sec\n",
        (double)((double)(circular_shift_end_cpu - circular_shift_start_cpu) / CLOCKS_PER_SEC));

    //check data of I5
    int rowI5 = I5[0].rows; // Number of rows in the image
    int colI5 = I5[0].cols; // Number of columns in the image
    cout << "I5's Last image's (row-1,col-1)/last pixel: " << checkPixels(I5[I5.size() - 1], rowI5 - 1, colI5 - 1) << "\n";//106

    I3.clear();
    I2.clear();
    I1.clear();
    I4.clear();
    I5.clear();
    cout << "Program End\n";
  
    return 0;
}
