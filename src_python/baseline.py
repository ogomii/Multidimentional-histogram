import cv2
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def histogram(sample, bins):
    N, D = sample.shape
    binsShape = [len(bin)-1 for bin in bins]
    counts = np.zeros(binsShape)
    for pixel in sample:
        binAffiliation = [binsShape[_]-1 for _ in range(D)]
        for colorIndex in range(D):
            for edgeIndex in range(binsShape[colorIndex]):
                if pixel[colorIndex] < bins[colorIndex][edgeIndex+1]:
                    binAffiliation[colorIndex] = edgeIndex
                    break
        counts[tuple(binAffiliation)] += 1
    return counts, bins

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='image path', required=True)
    parser.add_argument('--reduction_type', help='partial reduction type (columns or rows)', required=True)
    parser.add_argument('--dims', help='bins for each dimension', nargs='+', required=False, default=None)
    parser.add_argument('--seq', help='bins sequence', nargs='+', required=False, default=None)
    parser.add_argument('--save', help='save counts to file', required=False, action='store_true')
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

def getBinSequenceFromDimension(dims):
    dimenstions = [int(i) for i in dims]
    bins = [[0] for _ in range(len(dimenstions))]
    for dimIndex in range(len(dimenstions)):
        edge = 0
        for dim in range(dimenstions[dimIndex]):
            edge += 255/dimenstions[dimIndex]
            bins[dimIndex].append(edge)
    return bins

def calculateHistogramBasedOnBinsDimentaion(pixels, dims, verbose=True):
    bins = getBinSequenceFromDimension(dims)
    counts, bins = histogram(pixels, bins = bins)
    if verbose:
        print(f"histogram bin ranges:\n{bins[0]}\n{bins[1]}\n{bins[2]}")
        print(f"histogram output counts shape: {counts.shape}")
        print(f"counts of values in bins:\n{counts}")
    return counts, bins

def calculateHistogramBasedOnSequence(pixels, sequence, dimenstionality, verbose=True):
    binSequences = [[float(i) for i in sequence] for _ in range(dimenstionality)]
    counts, bins = np.histogramdd(pixels, bins = binSequences)
    if verbose:
        print(f"histogram bin ranges:\n{bins[0]}\n{bins[1]}\n{bins[2]}")
        print(f"histogram output counts shape: {counts.shape}")
        print(f"counts of values in bins:\n{counts}")
    return counts, bins

def calculate_histogram(data, dims=None, sequence=None):
    pixels = data.reshape(-1, data.shape[-1])

    if dims is not None:
        counts, bins = calculateHistogramBasedOnBinsDimentaion(pixels, dims)
    elif sequence is not None:
        counts, bins = calculateHistogramBasedOnSequence(pixels, sequence, pixels.shape[1])
    else:
        print("Bins unspecified!")
        exit()

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

def saveHistFromArray(array, fileName, hisogramIndex):
    with open(fileName, 'w') as outfile:
        for slice_2d in array[hisogramIndex]:
            np.savetxt(outfile, slice_2d)
    print(f"partial reduction histogram with index {hisogramIndex} saved to file {fileName}")

def rowBasedPartialReduction(image, bins):
    counts = []
    for row in image:
        row_counts, _ = histogram(row, bins)
        counts.append(row_counts)
    counts = np.array(counts)
    print(f"Partial reduction counts shape: {counts.shape}")
    saveHistFromArray(counts, "rowBasedPatialReduction.txt", 0)
    return counts

def columnBasedPartialReduction(image, bins):
    counts = []
    for i in range(image.shape[1]):  # Iterate over columns
        col_counts, _ = histogram(image[:, i], bins)  # Use image[:, i] instead of image[:, 0]
        counts.append(col_counts)
    counts = np.array(counts)
    print(f"Partial reduction counts shape: {counts.shape}")
    saveHistFromArray(counts, "columnsBasedPatialReduction.txt", 0)
    return counts

def partial_reduction(image, axis, bins):
    if axis == 'rows':
        counts = rowBasedPartialReduction(image, bins)
    elif axis == 'columns':
        counts = columnBasedPartialReduction(image, bins)
    else:
        raise ValueError("Invalid axis. Use 'rows' or 'columns'.")

    print(f"Partial reduction counts shape: {counts.shape}")
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

def saveCountsToFile(hist_counts):
    with open("pythonCounts.txt", "w") as txt_file:
        hist_counts_flattened = hist_counts.flatten('C')
        for count in hist_counts_flattened:
            txt_file.write(str(int(count)) + "\n")

if __name__ == "__main__":
    args = parseArguments()

    image = getImage(args.image)

    if image.shape[2] == 3:
        printExampleRGBValues(image)
    list_all_colors(image)


    hist_counts, hist_bins = calculate_histogram(image, args.dims, args.seq)
    if(args.save):
        saveCountsToFile(hist_counts)
    reduction_counts, reduction_bins = partial_reduction(image, axis=args.reduction_type, bins=hist_bins)

    #threshold = 100
    #remaining_bins = partial_decomposition(reduction_counts, reduction_bins, threshold)
    #print(f"Remaining bins after partial decomposition:\n{remaining_bins}")

    # Plot 3D histogram for histogram
    plot_3d_histogram(hist_counts, hist_bins)
