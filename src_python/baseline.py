import cv2
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='image path', required=True)
    parser.add_argument('--reduction_type', help='partial reduction type (columns or rows)', required=True)
    parser.add_argument('--dims', help='bins for each dimension', nargs='+', required=False, default=None)
    parser.add_argument('--ranges', help='bin ranges', nargs='+', required=False, default=None)
    return parser.parse_args()

def getImage(imagePath):
    try:
        image = cv2.imread(imagePath)
        if image is None:
            raise Exception("Loading image failed. Please check the image path.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"Image shape: {image.shape}")
        return image
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

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

def calculate_histogram(data, dims, ranges):
    pixels = data.reshape(-1, data.shape[-1])

    if dims is not None:
        # Use specified dimensions
        bins = tuple([dims[i] for i in range(min(len(dims), pixels.shape[-1]))])
    else:
        # Automatically determine dimensions
        if ranges is not None:
            bins = tuple([
                len(ranges[i]) - 1 if isinstance(ranges[i][0], (int, np.integer, float, np.floating)) else 256
                for i in range(len(ranges))
            ])
        else:
            bins = 16 # Default value

    # Ensure that ranges are of the same type for all dimensions
    if ranges is not None:
        ranges = tuple([
            (float(dim_range[0]), float(dim_range[1])) if isinstance(dim_range[0], (int, np.integer, float, np.floating))
            else dim_range
            for dim_range in ranges
        ])

    # Convert bins to a list of integers if it is a tuple containing integers and strings
    if isinstance(bins, tuple):
        bins = list(bins)
        for i in range(len(bins)):
            if isinstance(bins[i], str):
                bins[i] = int(bins[i])

    counts, bins = np.histogramdd(pixels, bins=bins, range=ranges)

    return counts, bins


def partial_decomposition(counts, bins, threshold):
    selected_bins = []
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            if np.any(counts[i, j] >= threshold):
                flat_index = i * counts.shape[1] + j
                if flat_index < len(bins):
                    selected_bins.append(bins[flat_index])

    selected_bins = np.array(selected_bins)

    # Check if selected_bins is not empty before performing the comparison
    if selected_bins.size > 0:
        remaining_bins = [
            bin_
            for bin_ in bins
            if not np.any(np.all(bin_ == selected_bins, axis=(0, 1), keepdims=True, where=~np.isnan(bin_)))
        ]
    else:
        remaining_bins = bins

    return remaining_bins

def partial_reduction(image, axis, dims, ranges):
    if axis == 'rows':
        # Perform partial reduction on rows
        counts = []
        for row in image:
            row_counts, _ = calculate_histogram(row, dims, ranges)
            counts.append(row_counts)
        counts = np.array(counts)

        # Aggregate counts along the specified axis (rows)
        counts = np.sum(counts, axis=0)

        # Extract bin edges for plotting
        _, bins = calculate_histogram(image[0], dims, ranges)

    elif axis == 'columns':
        # Perform partial reduction on columns
        counts = []
        for i in range(image.shape[1]):  # Iterate over columns
            col_counts, _ = calculate_histogram(image[:, i], dims, ranges)  # Use image[:, i] instead of image[:, 0]
            counts.append(col_counts)
        counts = np.array(counts)

        # Aggregate counts along the specified axis (columns)
        counts = np.sum(counts, axis=0)

        # Extract bin edges for plotting
        _, bins = calculate_histogram(image[:, 0], dims, ranges)  # Use image[:, 0] for bins

    else:
        raise ValueError("Invalid axis. Use 'rows' or 'columns'.")

    return counts, bins

def plot_3d_histogram(counts, bins):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x_data, y_data = np.meshgrid(bins[0], bins[1], indexing='ij')
    x_data = x_data.flatten()
    y_data = y_data.flatten()
    z_data = np.zeros_like(x_data)
    dx = np.diff(bins[0])[0]
    dy = np.diff(bins[1])[0]

    for i in range(len(x_data)):
        ax.bar3d(x_data[i], y_data[i], z_data[i], dx, dy, counts.flatten()[i], color='b', zsort='average')

    plt.show()

if __name__ == "__main__":
    args = parseArguments()

    image = getImage(args.image)

    if image.shape[2] == 3:
        printExampleRGBValues(image)
    list_all_colors(image)

    threshold = 100

    counts, bins = partial_reduction(image, axis=args.reduction_type, dims=args.dims, ranges=args.ranges)
    remaining_bins = partial_decomposition(counts, bins, threshold)

    print(f"Remaining bins after partial decomposition:\n{remaining_bins}")

    # Plot 3D histogram for partial reduction
    plot_3d_histogram(counts, bins)
