import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='image path', required=True)
    parser.add_argument('--dims', help='bins for each dimension', nargs='+', required=False, default=None)
    parser.add_argument('--ranges', help='bin ranges', nargs='+', required=False, default=None)
    return parser.parse_args()

def getImage(imagePath):
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"Image shape: {image.shape}")
    return image

def printExampleRGBValues(image):
    print("3 pixel values from the image in RGB:")
    (R, G, B) = image[0, 0]
    print(f"{R} {G} {B}")
    (R, G, B) = image[int(image.shape[0]/2), int(image.shape[1]/2)]
    print(f"{R} {G} {B}")
    (R, G, B) = image[int(image.shape[0]-1), int(image.shape[1]-1)]
    print(f"{R} {G} {B}")

def list_all_colors(image):
    unique_colors = np.unique(image.reshape(-1, image.shape[2]), axis=0)
    print(f"List of all unique colors in the image:\n{unique_colors}")

def calculate_histogram(image, dims, ranges):
    pixels = image.reshape(image.shape[0] * image.shape[1], image.shape[2])

    if dims is not None:
        counts, bins = np.histogramdd(pixels, bins=[int(i) for i in dims])
        print(f"Multidimensional histogram counts shape: {counts.shape}")
        print(f"Counts of values in bins:\n{counts}")
        return counts, bins
    elif ranges is not None:
        bin_ranges = [[float(i) for i in ranges] for _ in range(image.shape[2])]
        counts, bins = np.histogramdd(pixels, bins=bin_ranges)
        print(f"Histogram bin ranges:\n{bins[0]}\n{bins[1]}\n{bins[2]}")
        print(f"Counts of values in bins:\n{counts}")
        return counts, bins
    else:
        return np.array([]), []

def partial_decomposition(counts, bins, threshold):
    indices = np.where(counts > threshold)
    selected_bins = list(zip(*indices))

    print(f"Selected bins coordinates:\n{selected_bins}")

    selected_counts = counts[indices]
    remaining_bins = np.delete(bins, indices, axis=0)

    print(f"Selected counts:\n{selected_counts}")
    print(f"Remaining bins:\n{remaining_bins}")

def plot_3d_color_histogram(image):
    pixels = image.reshape(-1, image.shape[2])

    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)

    r_values = unique_colors[:, 0]
    g_values = unique_colors[:, 1]
    b_values = unique_colors[:, 2]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(r_values, g_values, b_values, 1, 1, counts, shade=True)

    ax.set_xlabel('Red Channel')
    ax.set_ylabel('Green Channel')
    ax.set_zlabel('Blue Channel')
    ax.set_title('3D Color Histogram')

    plt.show()

if __name__ == "__main__":
    args = parseArguments()

    image = getImage(args.image)

    if image.shape[2] == 3:
        printExampleRGBValues(image)
    list_all_colors(image)

    threshold = 100

    counts, bins = calculate_histogram(image, args.dims, args.ranges)
    partial_decomposition(counts, bins, threshold)

    plot_3d_color_histogram(image)
