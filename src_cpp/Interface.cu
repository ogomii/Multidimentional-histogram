#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

__global__ void histogramKernel(float* counts, const float* sample, const int* binsShape, const float* bins, const int D, const int totalBins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < totalBins) {
        int binAffiliation[3];
        int tempIdx = idx;
        for (int i = D - 1; i >= 0; --i) {
            binAffiliation[i] = tempIdx % binsShape[i];
            tempIdx /= binsShape[i];
        }

        bool inBin = true;
        for (int i = 0; i < D; ++i) {
            if (!(sample[i] >= bins[i * (binsShape[i] + 1) + binAffiliation[i]] - 1e-6 && 
                  sample[i] < bins[i * (binsShape[i] + 1) + binAffiliation[i] + 1] + 1e-6)) {
                inBin = false;
                break;
            }
        }

        if (inBin) {
            atomicAdd(&counts[idx], 1.0f);
        }
    }
}

std::vector<float> histogram(const std::vector<std::vector<float>>& sample, const std::vector<std::vector<float>>& bins) {
    const int N = sample.size();
    const int D = sample[0].size();

    std::vector<int> binsShape;
    for (const auto& bin : bins) {
        binsShape.push_back(bin.size() - 1);
    }

    const int totalBins = std::accumulate(binsShape.begin(), binsShape.end(), 1, std::multiplies<int>());

    float* d_sample;
    cudaMalloc((void**)&d_sample, N * D * sizeof(float));
    cudaMemcpy(d_sample, sample.data(), N * D * sizeof(float), cudaMemcpyHostToDevice);

    float* d_counts;
    cudaMalloc((void**)&d_counts, totalBins * sizeof(float));
    cudaMemset(d_counts, 0, totalBins * sizeof(float));

    int* d_binsShape;
    cudaMalloc((void**)&d_binsShape, binsShape.size() * sizeof(int));
    cudaMemcpy(d_binsShape, binsShape.data(), binsShape.size() * sizeof(int), cudaMemcpyHostToDevice);

    float* d_bins;
    cudaMalloc((void**)&d_bins, bins.size() * (D + 1) * sizeof(float));
    cudaMemcpy(d_bins, bins.data(), bins.size() * (D + 1) * sizeof(float), cudaMemcpyHostToDevice);

    const int blockSize = 256;
    const int gridSize = (totalBins + blockSize - 1) / blockSize;

    histogramKernel<<<gridSize, blockSize>>>(d_counts, d_sample, d_binsShape, d_bins, D, totalBins);

    std::vector<float> counts(totalBins);
    cudaMemcpy(counts.data(), d_counts, totalBins * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_sample);
    cudaFree(d_counts);
    cudaFree(d_binsShape);
    cudaFree(d_bins);

    return counts;
}

int main() {
    std::vector<std::vector<float>> sample = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
    std::vector<std::vector<float>> bins = {{0.0, 2.0, 4.0, 6.0}, {1.0, 3.0, 5.0, 7.0}, {2.0, 4.0, 6.0, 8.0}};

    std::vector<float> counts = histogram(sample, bins);

    for (float count : counts) {
        std::cout << count << " ";
    }

    return 0;
}