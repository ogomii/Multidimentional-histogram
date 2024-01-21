#include <iostream>
#include <vector>
#include <numeric>
#include <opencv2/opencv.hpp>

// CUDA kernel to calculate histogram
__global__ void histogramKernel(float* counts, const uchar3* image, const int* binsShape, const int width, const int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // Calculate bin affiliation for each color channel
        int binAffiliation[3];
        binAffiliation[0] = static_cast<int>(image[y * width + x].x);
        binAffiliation[1] = static_cast<int>(image[y * width + x].y);
        binAffiliation[2] = static_cast<int>(image[y * width + x].z);

        // Calculate the flat index for 3D histogram
        int flatIndex = binAffiliation[2] * binsShape[1] * binsShape[0] + binAffiliation[1] * binsShape[0] + binAffiliation[0];

        // Atomically increment the corresponding histogram bin
        atomicAdd(&counts[flatIndex], 1.0f);
    }
}

// Function to calculate histogram from an image
std::vector<float> histogram(const cv::Mat& image, const std::vector<int>& binsShape) {
    const int width = image.cols;
    const int height = image.rows;

    // Total number of bins in the histogram
    const int totalBins = std::accumulate(binsShape.begin(), binsShape.end(), 1, std::multiplies<int>());

    // Allocate device memory for histogram counts
    float* d_counts;
    cudaMalloc((void**)&d_counts, totalBins * sizeof(float));
    cudaMemset(d_counts, 0, totalBins * sizeof(float));

    // Copy image data to device
    uchar3* d_image;
    cudaMalloc((void**)&d_image, width * height * sizeof(uchar3));
    cudaMemcpy(d_image, image.ptr(), width * height * sizeof(uchar3), cudaMemcpyHostToDevice);

    // Copy binsShape to device
    int* d_binsShape;
    cudaMalloc((void**)&d_binsShape, binsShape.size() * sizeof(int));
    cudaMemcpy(d_binsShape, binsShape.data(), binsShape.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Define CUDA thread and block dimensions
    const dim3 blockSize(16, 16);
    const dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch CUDA kernel to calculate histogram
    histogramKernel<<<gridSize, blockSize>>>(d_counts, d_image, d_binsShape, width, height);

    // Copy histogram counts back to host
    std::vector<float> counts(totalBins);
    cudaMemcpy(counts.data(), d_counts, totalBins * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_counts);
    cudaFree(d_image);
    cudaFree(d_binsShape);

    return counts;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>\n";
        return -1;
    }

    // Load the image from the specified path in command line arguments
    cv::Mat image = cv::imread(argv[1]);
    if (image.empty()) {
        std::cerr << "Could not open or find the image.\n";
        return -1;
    }

    // Print image information
    std::cout << "Image size: " << image.cols << " x " << image.rows << std::endl;

    // Convert the image to uchar3 format
    cv::Mat3b bgrImage = image;
    std::vector<uchar3> uchar3Image;
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            cv::Vec3b pixel = bgrImage(y, x);
            uchar3Image.push_back(uchar3{pixel[0], pixel[1], pixel[2]});
        }
    }

    // Define histogram parameters (reduce the number of bins)
    std::vector<int> binsShape = {64, 64, 64};  // Adjust as needed

    // Calculate the histogram
    std::vector<float> counts = histogram(image, binsShape);

    // Display the results
    for (int i = 0; i < counts.size(); ++i) {
        if (counts[i] > 0) {
            std::cout << "Bin " << i << ": " << counts[i] << std::endl;
        }
    }

    return 0;
}
