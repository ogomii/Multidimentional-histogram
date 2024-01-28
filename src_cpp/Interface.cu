#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

__device__ int getBinIndex(int value, const int* bins, int binSize) {
    for (int i = 0; i < binSize - 1; ++i) {
        if (value < bins[i * 3 + 1]) {
            return i;
        }
    }
    return binSize - 1;
}

__global__ void histogramKernel(const uchar3* inputImage, int width, int height, const int* bins, int* counts, int binSize) {
    int tidX = blockIdx.x * blockDim.x + threadIdx.x;
    int tidY = blockIdx.y * blockDim.y + threadIdx.y;

    if (tidX < width && tidY < height) {
        uchar3 pixel = inputImage[tidY * width + tidX];

        // Calculate bin indices for each channel
        int binIndexR = getBinIndex(pixel.x, bins, binSize);
        int binIndexG = getBinIndex(pixel.y, bins, binSize);
        int binIndexB = getBinIndex(pixel.z, bins, binSize);

        // Increment the corresponding bin in 1D counts array
        int countsIndex = binIndexR * binSize * binSize + binIndexG * binSize + binIndexB;
        int pixelIndex = tidY * width + tidX;
        atomicAdd(&counts[countsIndex], (pixelIndex < width * height) ? 1 : 0);
    }
}

void calculateHistogram(const uchar3* hostImage, int width, int height, const int* hostBins, int* hostCounts, int binSize) {
    // Allocate device memory
    uchar3* deviceImage;
    int* deviceBins;
    int* deviceCounts;

    cudaMalloc(&deviceImage, width * height * sizeof(uchar3));
    cudaMalloc(&deviceBins, 3 * binSize * sizeof(int));
    cudaMalloc(&deviceCounts, binSize * binSize * binSize * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(deviceImage, hostImage, width * height * sizeof(uchar3), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceBins, hostBins, 3 * binSize * sizeof(int), cudaMemcpyHostToDevice);

    // Set block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // Launch the kernel
    histogramKernel<<<gridDim, blockDim>>>(deviceImage, width, height, deviceBins, deviceCounts, binSize);

    // Copy result back to host
    cudaMemcpy(hostCounts, deviceCounts, binSize * binSize * binSize * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(deviceImage);
    cudaFree(deviceBins);
    cudaFree(deviceCounts);
}

int main(int argc, char** argv) {
    // Example usage
    const int binSize = 3;

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_image>" << std::endl;
        return -1;
    }

    // Read the image
    cv::Mat inputImage = cv::imread(argv[1]);
    if (inputImage.empty()) {
        std::cerr << "Error: Could not read the image." << std::endl;
        return -1;
    }

    const int width = inputImage.cols;
    const int height = inputImage.rows;

    uchar3* image = new uchar3[width * height];
    int* bins = new int[3 * binSize];
    int* counts = new int[binSize * binSize * binSize];

    // Initialize image and bins
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            cv::Vec3b pixel = inputImage.at<cv::Vec3b>(y, x);

            // Assuming uchar3 for image representation
            image[y * width + x] = make_uchar3(pixel[2], pixel[1], pixel[0]);
        }
    }

    // Initialize bins (replace with your own bin initialization logic)
    for (int i = 0; i < binSize; ++i) {
        bins[i * 3 + 0] = 0;
        bins[i * 3 + 1] = 125;
        bins[i * 3 + 2] = 255;
    }

    // Calculate histogram
    calculateHistogram(image, width, height, bins, counts, binSize);

    // Print or use the result as needed
    std::cout << "Histogram Counts:" << std::endl;
    for (int i = 0; i < binSize * binSize * binSize; ++i) {
        std::cout << counts[i] << " ";
        if ((i + 1) % binSize == 0) {
            std::cout << std::endl;
        }
    }

    // Cleanup
    delete[] image;
    delete[] bins;
    delete[] counts;

    return 0;
}