#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

// Kernel CUDA pour le calcul de l'histogramme
__global__ void histogramKernel(const uchar* d_data, int width, int height, int channels,
                                int* d_counts, const int* d_bins, int binsSize) {
    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    int tid_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (tid_x < width && tid_y < height) {
        int index = tid_y * width * channels + tid_x * channels;

        for (int c = 0; c < channels; ++c) {
            int pixelValue = d_data[index + c];
            int binIndex = binsSize - 1;

            for (int i = 0; i < binsSize - 1; ++i) {
                if (pixelValue < d_bins[c * binsSize + i + 1]) {
                    binIndex = i;
                    break;
                }
            }

            atomicAdd(&d_counts[c * binsSize * binsSize + binIndex * binsSize + c], 1);
        }
    }
}

// Fonction pour calculer l'histogramme en couleur à l'aide de CUDA
void calculateColorHistogram(const cv::Mat& image, const std::vector<std::vector<int>>& bins,
                             std::vector<int>& counts) {
    int width = image.cols;
    int height = image.rows;
    int channels = image.channels();
    int binsSize = bins[0].size();

    uchar* d_data;
    int* d_counts;
    int* d_bins;

    // Allouer de la mémoire sur le GPU
    cudaMalloc((void**)&d_data, width * height * channels * sizeof(uchar));
    cudaMalloc((void**)&d_counts, channels * binsSize * binsSize * sizeof(int));
    cudaMalloc((void**)&d_bins, channels * binsSize * sizeof(int));

    // Copier les données sur le GPU
    cudaMemcpy(d_data, image.data, width * height * channels * sizeof(uchar), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bins, bins.data(), channels * binsSize * sizeof(int), cudaMemcpyHostToDevice);

    // Définir la grille et les blocs pour l'exécution du kernel CUDA
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Appeler le kernel CUDA
    histogramKernel<<<numBlocks, threadsPerBlock>>>(d_data, width, height, channels, d_counts, d_bins, binsSize);

    // Copier les résultats du GPU vers le CPU
    cudaMemcpy(counts.data(), d_counts, channels * binsSize * binsSize * sizeof(int), cudaMemcpyDeviceToHost);

    // Libérer la mémoire sur le GPU
    cudaFree(d_data);
    cudaFree(d_counts);
    cudaFree(d_bins);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    // Charger l'image en couleur (3 canaux)
    cv::Mat image = cv::imread(argv[1]);

    if (image.empty()) {
        std::cerr << "Error: Could not read the image." << std::endl;
        return -1;
    }

    // Définir les bins
    std::vector<std::vector<int>> bins = {{0, 125, 255}, {0, 125, 255}, {0, 125, 255}};

    // Initialiser le vecteur de counts avec des zéros
    std::vector<int> counts(bins[0].size() * bins[0].size() * bins[0].size(), 0);

    // Calculer l'histogramme en couleur à l'aide de CUDA
    calculateColorHistogram(image, bins, counts);

    // Afficher les résultats
    std::cout << "Color Histogram Counts:" << std::endl;
    for (int i = 0; i < bins[0].size(); ++i) {
        for (int j = 0; j < bins[0].size(); ++j) {
            for (int k = 0; k < bins[0].size(); ++k) {
                std::cout << counts[i * bins[0].size() * bins[0].size() + j * bins[0].size() + k] << " ";
            }
            std::cout << std::endl;
        }
    }

    return 0;
}
