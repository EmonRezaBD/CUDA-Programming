#include "cuda_runtime.h"
#include "device_launch_parameters.h"


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

using namespace std;
using namespace cv;
using namespace cv::cuda;

#define PI 3.14159265358979323846f
#define LAMBDA_EQUIV 2.64f  // Equivalent wavelength in micrometers

// Load grayscale stack and Z-positions
bool loadStackAndZ(const string& dir, int numImages, vector<GpuMat>& gpuStack, vector<float>& zValues) {
    gpuStack.clear();
    zValues.clear();

    // Load Z values from CSV
    ifstream file(dir + "/bump_img_wli_raft.CSV");
    if (!file.is_open()) {
        //std::cerr << "Error opening file!" << std::endl;
        return false;
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
                    zValues.push_back(std::stod(cell));
                    /*sAmp.PzPos_um = columnData[cnt];
                    ImgInfo.push_back(sAmp);*/
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

    if (zValues.size() != numImages) return false;

    // Load grayscale images to GPU
    for (int i = 1; i <= numImages; ++i) {
        stringstream path;
        path << dir << "bump_img_wli_raft_" << i << ".BMP";
        Mat img = imread(path.str(), IMREAD_COLOR);
        if (img.empty()) return false;

        Mat gray;
        cv::cvtColor(img, gray, COLOR_BGR2GRAY);
        GpuMat gGray;
        gGray.upload(gray);
        gpuStack.push_back(gGray);
    }

    return true;
}

// CUDA Kernel: compute phase per pixel
__global__ void computePhaseKernel(const uchar** images, const float* sinPhi, const float* cosPhi,
    float* phaseOut, int width, int height, int numFrames, size_t step) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float sumSin = 0.0f;
    float sumCos = 0.0f;
    for (int i = 0; i < numFrames; ++i) {
        const uchar* img = images[i];
        int idx = y * step + x;
        float intensity = static_cast<float>(img[idx]);
        sumSin += intensity * sinPhi[i];
        sumCos += intensity * cosPhi[i];
    }

    int outIdx = y * width + x;
    phaseOut[outIdx] = atan2f(sumSin, sumCos);
}

void computePhaseMap(const vector<GpuMat>& gpuStack, const vector<float>& zVals, GpuMat& phaseMap) {
    int numFrames = gpuStack.size();
    int width = gpuStack[0].cols;
    int height = gpuStack[0].rows;

    // Step 1: Precompute sin/cos(2π * Z / λ)
    vector<float> sinPhi(numFrames), cosPhi(numFrames);
    for (int i = 0; i < numFrames; ++i) {
        float phi = 2.0f * PI * zVals[i] / LAMBDA_EQUIV;
        sinPhi[i] = sinf(phi);
        cosPhi[i] = cosf(phi);
    }

    // Step 2: Upload sin/cos arrays to GPU
    GpuMat d_sinPhi(numFrames, 1, CV_32F), d_cosPhi(numFrames, 1, CV_32F);
    d_sinPhi.upload(Mat(sinPhi).reshape(1, numFrames));
    d_cosPhi.upload(Mat(cosPhi).reshape(1, numFrames));

    // Step 3: Create host array of image pointers
    vector<const uchar*> imgPtrs(numFrames);
    for (int i = 0; i < numFrames; ++i)
        imgPtrs[i] = gpuStack[i].ptr<uchar>();

    // Step 4: Upload image pointers to device
    const uchar** d_imgPtrs;
    cudaMalloc(&d_imgPtrs, numFrames * sizeof(uchar*));
    cudaMemcpy(d_imgPtrs, imgPtrs.data(), numFrames * sizeof(uchar*), cudaMemcpyHostToDevice);

    // Step 5: Allocate output
    GpuMat d_phase(height, width, CV_32F);

    // Step 6: Launch kernel
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    computePhaseKernel << <grid, block >> > (
        d_imgPtrs, d_sinPhi.ptr<float>(), d_cosPhi.ptr<float>(),
        d_phase.ptr<float>(), width, height, numFrames, gpuStack[0].step
        );

    // Step 7: Error check and sync
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "CUDA kernel failed: " << cudaGetErrorString(err) << endl;
        cudaFree(d_imgPtrs);
        return;
    }
    cudaDeviceSynchronize();

    // Step 8: Output result
    phaseMap = d_phase.clone();

    // Step 9: Cleanup
    cudaFree(d_imgPtrs);
}

void savePhaseToCSV(const Mat& phaseImage, const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open CSV file for writing!" << endl;
        return;
    }

    for (int i = 0; i < phaseImage.rows; ++i) {
        for (int j = 0; j < phaseImage.cols; ++j) {
            float phaseVal = phaseImage.at<float>(i, j);
            file << phaseVal << ",";
            //if (j != img.cols - 1) file << ", ";  // Avoid comma at the end of the line
        }
        file << "\n";  // Newline for the next row
    }

    /*for (int y = 0; y < phaseImage.rows; ++y) {
        for (int x = 0; x < phaseImage.cols; ++x) {
            float phaseVal = phaseImage.at<float>(y, x);
            file << x << "," << y << "," << phaseVal << "\n";
        }
    }*/

    file.close();
    cout << "Saved CSV: " << filename << endl;
}

int main() {
    const string imageDir = "E:\\Interferometry\\RAFT\\bump_img_wli_raft\\";
    const int numFrames = 415;

    vector<GpuMat> gpuStack;
    vector<float> zVals;

    if (!loadStackAndZ(imageDir, numFrames, gpuStack, zVals)) {
        cerr << "Error loading image stack or Z values." << endl;
        return -1;
    }

    GpuMat phaseMap;
    computePhaseMap(gpuStack, zVals, phaseMap);

    Mat cpuPhase;
    phaseMap.download(cpuPhase);
    savePhaseToCSV(cpuPhase, "phase_map.csv");

    Mat normalized;
    normalize(cpuPhase, normalized, 0, 255, NORM_MINMAX);
    normalized.convertTo(normalized, CV_8U);
    imwrite("phase_result.png", normalized);

    cout << "Phase map generated and saved as 'phase_result.png' and 'phase_map.csv'" << endl;

    return 0;
}