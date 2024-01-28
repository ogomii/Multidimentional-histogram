#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

constexpr int getCountsSize(int binSize)
{
    return std::pow(binSize-1,3);
}

__device__ int getBinIndex(int value, const float* bins, int binCountPerDim) {
    for (int i = 0; i < binCountPerDim; ++i) {
        if (value < bins[i + 1]) {
            return i;
        }
    }
    return binCountPerDim-1;
}

__global__ void histogramKernel(const uchar3* inputImage, int width, int height, const float* bins, int* counts, int binCountPerDim) {
    int tidX = blockIdx.x * blockDim.x + threadIdx.x;
    int tidY = blockIdx.y * blockDim.y + threadIdx.y;

    if (tidX < width && tidY < height) {
        uchar3 pixel = inputImage[tidY * width + tidX];

        int binIndexR = getBinIndex(pixel.x, &bins[0*(binCountPerDim+1)], binCountPerDim);
        int binIndexG = getBinIndex(pixel.y, &bins[1*(binCountPerDim+1)], binCountPerDim);
        int binIndexB = getBinIndex(pixel.z, &bins[2*(binCountPerDim+1)], binCountPerDim);

        int countsIndex = binIndexR * binCountPerDim * binCountPerDim + binIndexG * binCountPerDim + binIndexB;
        int pixelIndex = tidY * width + tidX;
        atomicAdd(&counts[countsIndex], (pixelIndex < width * height) ? 1 : 0);
    }
}

void calculateHistogram(const uchar3* hostImage, int width, int height, const float* hostBins, int* hostCounts, int binSize) {
    uchar3* deviceImage;
    float* deviceBins;
    int* deviceCounts;

    cudaMalloc(&deviceImage, width * height * sizeof(uchar3));
    cudaMalloc(&deviceBins, 3 * binSize * sizeof(float));
    cudaMalloc(&deviceCounts, getCountsSize(binSize) * sizeof(int));

    cudaMemcpy(deviceImage, hostImage, width * height * sizeof(uchar3), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceBins, hostBins, 3 * binSize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(8,8);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    auto start = std::chrono::high_resolution_clock::now();
    histogramKernel<<<gridDim, blockDim>>>(deviceImage, width, height, deviceBins, deviceCounts, (binSize-1));
    cudaDeviceSynchronize();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;

    cudaMemcpy(hostCounts, deviceCounts, getCountsSize(binSize) * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(deviceImage);
    cudaFree(deviceBins);
    cudaFree(deviceCounts);
}

int main(int argc, char** argv) {
    const int binSize = 3;

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_image>" << std::endl;
        return -1;
    }

    cv::Mat inputImage = cv::imread(argv[1]);
    if (inputImage.empty()) {
        std::cerr << "Error: Could not read the image." << std::endl;
        return -1;
    }

    const int width = inputImage.cols;
    const int height = inputImage.rows;

    uchar3* image = new uchar3[width * height];
    float bins[3 * binSize] = {0, 99.5, 255, 0, 99.5, 255, 0, 99.5, 255};
    int* counts = new int[getCountsSize(binSize)];

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            cv::Vec3b pixel = inputImage.at<cv::Vec3b>(y, x);

            // Assuming uchar3 for image representation
            image[y * width + x] = make_uchar3(pixel[2], pixel[1], pixel[0]);
        }
    }

    calculateHistogram(image, width, height, bins, counts, binSize);

    std::cout << "Histogram Counts:" << std::endl;
    for (int i = 0; i < getCountsSize(binSize); ++i) {
        std::cout << counts[i] << " ";
        if ((i + 1) % (binSize-1) == 0) {
            std::cout << std::endl;
        }
    }

    std::ofstream myfile ("cudaCounts.txt");
    if (myfile.is_open())
    {
        for (int i = 0; i < getCountsSize(binSize); ++i) {
            myfile << counts[i] << "\n";
        }
        myfile.close();
    }

    delete[] image;
    delete[] counts;

    return 0;
}