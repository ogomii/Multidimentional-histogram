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

def calculate_histogram(image, dims=None, ranges=None):
    pixels = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
    counts, bins = np.array([]), []

    try:
        if dims is not None:
            # Check if the number of specified dimensions matches the actual number of dimensions
            if len(dims) == pixels.shape[1]:
                # Calculate histogram based on dimensions
                counts, bins = np.histogramdd(pixels, bins=[int(i) for i in dims])
                print(f"Multidimensional histogram counts (dims) shape: {counts.shape}")
                print(f"Counts of values in bins (dims):\n{counts}")
            else:
                raise ValueError(f"Number of dimensions specified in dims ({len(dims)}) does not match the actual number of dimensions ({pixels.shape[1]})")

        if ranges is not None:
            # Calculate histogram based on ranges
            bin_ranges = [[float(i) for i in ranges] for _ in range(pixels.shape[1])]
            counts, bins = np.histogramdd(pixels, bins=bin_ranges)
            print(f"Histogram bin ranges:\n{bins[0]}\n{bins[1]}\n{bins[2]}")
            print(f"Counts of values in bins (ranges):\n{counts}")

    except ValueError as ve:
        print(f"Error: {ve}")

    return counts, bins

def partial_decomposition(counts, bins, threshold):
    indices = np.where(counts > threshold)
    selected_bins = list(zip(*indices))

    print(f"Selected bins coordinates:\n{selected_bins}")

    selected_counts = counts[indices]
    remaining_bins = [bin_ for bin_ in bins if not np.all(bin_[:3] == selected_bins, axis=1).any()]

    print(f"Selected counts:\n{selected_counts}")
    print(f"Remaining bins:\n{remaining_bins}")
    return remaining_bins

def partial_reduction(image, reduction_type):
    if reduction_type == 'columns':
        counts, bins = np.histogramdd(image.reshape(-1, 3), bins=[image.shape[0], 4, 4])
        reduced_counts = np.sum(counts, axis=0)
        print(f"Reduced counts along columns:\n{reduced_counts}")

    elif reduction_type == 'rows':
        counts, bins = np.histogramdd(image.reshape(-1, 3), bins=[image.shape[1], 4, 4])
        reduced_counts = np.sum(counts, axis=1)
        print(f"Reduced counts along rows:\n{reduced_counts}")

    else:
        print("Invalid reduction type. Please use 'columns' or 'rows'.")

    return reduced_counts, bins


# def partial_histogram(image, reduction_type):
#     histograms = []

#     if reduction_type == 'columns':
#         for col in range(image.shape[1]):
#             column_pixels = image[:, col, :]
#             counts, bins = np.histogramdd(column_pixels, bins=[4, 4, 4])
#             histograms.append((counts, bins))

#     elif reduction_type == 'rows':
#         for row in range(image.shape[0]):
#             row_pixels = image[row, :, :]
#             counts, bins = np.histogramdd(row_pixels, bins=[4, 4, 4])
#             histograms.append((counts, bins))

#     else:
#         print("Invalid reduction type. Please use 'columns' or 'rows'.")

#    return histograms

def plot_3d_histogram(counts, bins):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Extract bin edges
    x_edges, y_edges, z_edges = bins

    # Create grid
    x_data, y_data, z_data = np.meshgrid(x_edges[:-1], y_edges[:-1], z_edges[:-1], indexing='ij')

    # Flatten counts and coordinates
    flat_counts = counts.flatten()
    flat_x = x_data.flatten()
    flat_y = y_data.flatten()
    flat_z = z_data.flatten()

    # Reshape arrays to have the same size
    size = min(flat_counts.size, flat_x.size, flat_y.size, flat_z.size)
    flat_counts = flat_counts[:size]
    flat_x = flat_x[:size]
    flat_y = flat_y[:size]
    flat_z = flat_z[:size]

    # Plot 3D histogram
    ax.bar3d(flat_x, flat_y, flat_z, dx=1, dy=1, dz=flat_counts, zsort='average')

    ax.set_xlabel('Red Channel')
    ax.set_ylabel('Green Channel')
    ax.set_zlabel('Blue Channel')
    ax.set_title('3D Histogram')

    plt.show()

if __name__ == "__main__":
    args = parseArguments()

    image = getImage(args.image)

    if image.shape[2] == 3:
        printExampleRGBValues(image)
    list_all_colors(image)

    threshold = 100

    counts, bins = calculate_histogram(image, args.dims, args.ranges)
    remaining_bins = partial_decomposition(counts, bins, threshold)

    print(f"Remaining bins after partial decomposition:\n{remaining_bins}")

    # Partial reduction
    reduced_counts, reduced_bins = partial_reduction(image, args.reduction_type)

    # Plot 3D histogram for partial reduction
    plot_3d_histogram(reduced_counts, reduced_bins)
