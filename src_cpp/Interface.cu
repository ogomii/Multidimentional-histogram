#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

struct Metadata
{
    int dim;
    int width;
    int height;
};

constexpr int getCountsSize(int binSize, int dim)
{
    return std::pow(binSize-1,dim);
}

template<typename inputDataType>
__device__ int getBinIndex(inputDataType value, const float* bins, int binCountPerDim) {
    for (int i = 0; i < binCountPerDim; ++i) {
        if (static_cast<float>(value) < bins[i + 1])
        {
            return i;
        }
    }
    return binCountPerDim-1;
}

template<typename inputDataType>
__global__ void histogramKernel(const inputDataType* inputImage, const Metadata metadata, const float* bins, int* counts, const int binCountPerDim) {
    int tidX = blockIdx.x * blockDim.x + threadIdx.x;
    int tidY = blockIdx.y * blockDim.y + threadIdx.y;

    if (tidX < metadata.width && tidY < metadata.height) {
        const inputDataType* pixel = &inputImage[tidY * metadata.width * metadata.dim + tidX * metadata.dim];
        int weight = 1;
        int countsIndex = 0;
        for(int dimIndex = metadata.dim-1; dimIndex >= 0; dimIndex--)
        {
            countsIndex += weight * getBinIndex<inputDataType>(pixel[dimIndex], &bins[dimIndex*(binCountPerDim+1)], binCountPerDim);
            weight *= binCountPerDim;
        }
        atomicAdd(&counts[countsIndex], 1);
    }
}

template<typename inputDataType>
void calculateHistogram(const inputDataType* hostImage, const Metadata metadata, const float* hostBins, int* hostCounts, int binSize) {
    inputDataType* deviceImage;
    float* deviceBins;
    int* deviceCounts;

    cudaMalloc(&deviceImage, metadata.width * metadata.height * metadata.dim * sizeof(inputDataType));
    cudaMalloc(&deviceBins, metadata.dim * binSize * sizeof(float));
    cudaMalloc(&deviceCounts, getCountsSize(binSize, metadata.dim) * sizeof(int));

    cudaMemcpy(deviceImage, hostImage, metadata.width * metadata.height * metadata.dim * sizeof(inputDataType), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceBins, hostBins, metadata.dim * binSize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(8,8);
    dim3 gridDim((metadata.width + blockDim.x - 1) / blockDim.x, (metadata.height + blockDim.y - 1) / blockDim.y);

    auto start = std::chrono::high_resolution_clock::now();
    histogramKernel<inputDataType><<<gridDim, blockDim>>>(deviceImage, metadata, deviceBins, deviceCounts, (binSize-1));
    cudaDeviceSynchronize();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken by histogram kernel: " << duration.count() << " microseconds" << std::endl;

    cudaMemcpy(hostCounts, deviceCounts, getCountsSize(binSize, metadata.dim) * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(deviceImage);
    cudaFree(deviceBins);
    cudaFree(deviceCounts);
}


template<typename inputDataType>
__global__ void partialDecompositionKernel(const inputDataType* inputImage, const int row, const Metadata metadata, const float* bins, int* counts, const int binCountPerDim) {
    int tidX = blockIdx.x * blockDim.x + threadIdx.x;
    const int imageRow = row * metadata.width * metadata.dim;
    const int countsRow = row * std::pow(binCountPerDim, metadata.dim);

    if (tidX < metadata.width) {
        const inputDataType* pixel = &inputImage[imageRow + tidX * metadata.dim];
        int weight = 1;
        int countsIndex = 0;
        for(int dimIndex = metadata.dim-1; dimIndex >= 0; dimIndex--)
        {
            countsIndex += weight * getBinIndex<inputDataType>(pixel[dimIndex], &bins[dimIndex*(binCountPerDim+1)], binCountPerDim);
            weight *= binCountPerDim;
        }
        atomicAdd(&counts[countsRow + countsIndex], 1);
    }
}

template<typename inputDataType>
void calculateRowPartialReduction(const inputDataType* hostImage, const Metadata metadata, const float* hostBins, int* hostCounts, int binSize) {
    inputDataType* deviceImage;
    float* deviceBins;
    int* deviceCounts;
    int numOfBlocksPerRow = std::ceil(metadata.width / 1024.0);
    int numOfThreadsPerBlock = metadata.width / numOfBlocksPerRow;

    cudaMalloc(&deviceImage, metadata.width * metadata.height * metadata.dim * sizeof(inputDataType));
    cudaMalloc(&deviceBins, metadata.dim * binSize * sizeof(float));
    cudaMalloc(&deviceCounts, metadata.height * getCountsSize(binSize, metadata.dim) * sizeof(int));

    cudaMemcpy(deviceImage, hostImage, metadata.width * metadata.height * metadata.dim * sizeof(inputDataType), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceBins, hostBins, metadata.dim * binSize * sizeof(float), cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();

    for(int row = 0; row < metadata.height; row++)
    {
        partialDecompositionKernel<inputDataType><<<numOfBlocksPerRow, numOfThreadsPerBlock>>>(deviceImage, row, metadata, deviceBins, deviceCounts, (binSize-1));
    }

    cudaDeviceSynchronize();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken by partial reduction: " << duration.count() << " microseconds" << std::endl;

    cudaMemcpy(hostCounts, deviceCounts, metadata.height * getCountsSize(binSize, metadata.dim) * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(deviceImage);
    cudaFree(deviceBins);
    cudaFree(deviceCounts);
}

int main(int argc, char** argv) {

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_image>" << std::endl;
        return -1;
    }

    cv::Mat inputImage = cv::imread(argv[1]);
    if (inputImage.empty()) {
        std::cerr << "Error: Could not read the image." << std::endl;
        return -1;
    }

    Metadata metadata {.dim=3,
                       .width=inputImage.cols,
                       .height=inputImage.rows};
    std::cout << "Metadata height: " << metadata.height << " width: " << metadata.width << " dim: " << metadata.dim << std::endl;
    const int binSize = 5;
    using inputDataType = uint8_t;
    inputDataType* image = new inputDataType[metadata.width * metadata.height * metadata.dim];

    float bins[binSize * metadata.dim] = {0, 63.75, 127.5, 191.25, 255.0,
                                          0, 63.75, 127.5, 191.25, 255.0,
                                          0, 63.75, 127.5, 191.25, 255.0};
    int* counts = new int[getCountsSize(binSize, metadata.dim)];
    for (int y = 0; y < metadata.height; ++y) {
        for (int x = 0; x < metadata.width; ++x) {
            cv::Vec3b pixel = inputImage.at<cv::Vec3b>(y, x);

            // Assuming 1d array for image representation
            image[y * metadata.width * metadata.dim + x * metadata.dim + 0] = static_cast<inputDataType>(pixel[2]);
            image[y * metadata.width * metadata.dim + x * metadata.dim + 1] = static_cast<inputDataType>(pixel[1]);
            image[y * metadata.width * metadata.dim + x * metadata.dim + 2] = static_cast<inputDataType>(pixel[0]);
        }
    }

    calculateHistogram<inputDataType>(image, metadata, bins, counts, binSize);
    std::cout << "Histogram Counts:" << std::endl;
    for (int i = 0; i < getCountsSize(binSize, metadata.dim); ++i) {
        std::cout << counts[i] << " ";
        if ((i + 1) % (binSize-1) == 0)
        {
            std::cout << std::endl;
        }
    }
    std::ofstream myfile ("cudaCounts.txt");
    if (myfile.is_open())
    {
        for (int i = 0; i < getCountsSize(binSize, metadata.dim); ++i) {
            myfile << counts[i] << "\n";
        }
        myfile.close();
    }

    int* countsPartialReduction = new int[metadata.height * getCountsSize(binSize, metadata.dim)];
    calculateRowPartialReduction<inputDataType>(image, metadata, bins, countsPartialReduction, binSize);

    std::cout << "First row Counts:" << std::endl;
    int rowHistogramToPrint = 0;
    for (int i = 0; i < getCountsSize(binSize, metadata.dim); ++i) {
        std::cout << countsPartialReduction[rowHistogramToPrint * getCountsSize(binSize, metadata.dim) + i] << " ";
        if ((i + 1) % (binSize-1) == 0)
        {
            std::cout << std::endl;
        }
    }

    delete[] image;
    delete[] counts;
    delete[] countsPartialReduction;

    return 0;
}