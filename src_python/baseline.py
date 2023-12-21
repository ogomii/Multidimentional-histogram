import cv2
import argparse
import numpy as np

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='image path', required=True)
    parser.add_argument('--dims', help='bins for each dimention', nargs='+', required=False, default=None)
    parser.add_argument('--ranges', help='bin ranges', nargs='+', required=False, default=None)
    return parser.parse_args()

def getImage(imagePath):
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"image shape: {image.shape}")
    return image

def printExampleRGBValues(image):
    print("3 pixel value from the image in rgb:")
    (R, G, B) = image[0, 0]
    print(f"{R} {G} {B}")
    (R, G, B) = image[int(image.shape[0]/2), int(image.shape[1]/2)]
    print(f"{R} {G} {B}")
    (R, G, B) = image[int(image.shape[0]-1), int(image.shape[1]-1)]
    print(f"{R} {G} {B}")

def calculateHistogramBasedOnBinsDimentaion(pixels, dims):
    counts, bins = np.histogramdd(pixels, bins = [int(i) for i in dims])
    print(f"histogram output counts shape: {counts.shape}")
    print(f"counts of values in bins:\n{counts}")

def calculateHistogramBasedOnBinsRanges(pixels, ranges, dimenstionality):
    binRanges = [[float(i) for i in ranges] for _ in range(dimenstionality)]
    counts, bins = np.histogramdd(pixels, bins = binRanges)
    print(f"histogram bin ranges:\n{bins[0]}\n{bins[1]}\n{bins[2]}")
    print(f"counts of values in bins:\n{counts}")


if __name__ == "__main__":
    args = parseArguments()

    image = getImage(args.image)

    if image.shape[2] == 3:
        printExampleRGBValues(image)

    pixels = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
    print(f"shape of the histogram input array: {pixels.shape}")

    if args.dims != None:
        calculateHistogramBasedOnBinsDimentaion(pixels, args.dims)

    if args.ranges != None:
        calculateHistogramBasedOnBinsRanges(pixels, args.ranges, image.shape[2])