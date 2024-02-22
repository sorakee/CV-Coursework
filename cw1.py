import sys
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

INPUT_IMAGE = "./kitty.bmp"

WINDOW_WIDTH = 300
WINDOW_HEIGHT = 300

# Define kernels
SOBEL_X  = np.array([[-1,  0,  1],
                     [-2,  0,  2],
                     [-1,  0,  1]])
SOBEL_Y  = np.array([[-1, -2, -1],
                     [ 0,  0,  0],
                     [ 1,  2,  1]])

# Convolve image given kernel
def convolution(img, kernel):
    result = np.zeros(img.shape)

    kernel_w = kernel.shape[0]
    kernel_h = kernel.shape[1]

    pad = kernel_w // 2
    padded_img = padding(img, pad, pad)

    # Iterate over each pixel in the image
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel = padded_img[i:i + kernel_w, j:j + kernel_h]
            result[i, j] = np.sum(pixel * kernel)

    return result

# Generate gaussian kernel given size and sigma value
def gaussian(size, sigma=1):
    # Generate numpy array of indices 
    # and subtract each element by the center value
    x, y = np.indices((size, size)) - size // 2

    # G(x, y) = e^-((x^2+y^2)/(2*sigma^2))
    result = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    return result

# Generate mean/box kernel given size
def mean(size):
    result = np.ones((size, size), dtype=np.uint8)
    return result / (size ** 2)

# Padding Function
def padding(img, pad_w, pad_h):
    r, c = img.shape

    # Initialize result (padded image) with zeros
    result = np.zeros((r + 2 * pad_h, c + 2 * pad_w), dtype=img.dtype)
    result[pad_h:pad_h + r, pad_w:pad_w + c] = img

    return result

# Binary Threshold Function
def binary_threshold(img, threshold):
    # Initialize result with 0s
    result = np.zeros(img.shape, dtype=img.dtype)
    
    # Iterate over each pixel in the image
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            result[i, j] = 255 if img[i, j] >= threshold else 0
    
    return result

def thresholding_mean(threshold):
    _, result = cv.threshold(normalise(edge_strength_mean), threshold, 255, cv.THRESH_BINARY)
    cv.imshow("Threshold - Mean", result)

def thresholding_gaussian(threshold):
    _, result = cv.threshold(normalise(edge_strength_gaussian), threshold, 255, cv.THRESH_BINARY)
    cv.imshow("Threshold - Gaussian", result)

def thresholding_grey(threshold):
    _, result = cv.threshold(normalise(edge_strength_grey), threshold, 255, cv.THRESH_BINARY)
    cv.imshow("Threshold - Grey", result)

# Normalise values to range [0, 255]
def normalise(img):
    result = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255)
    return result.astype(np.uint8)

# Calculate gradient magnitude
def gradient_magnitude(sobelX, sobelY):
    return np.sqrt(sobelX ** 2 + sobelY ** 2)

def main():
    global edge_strength_mean
    global edge_strength_gaussian
    global edge_strength_grey

    m_img = cv.imread(INPUT_IMAGE)

    if m_img is None:
        print("ERROR: Failed to open image")
        sys.exit()

    grey_img = cv.cvtColor(m_img, cv.COLOR_BGR2GRAY)
    cv.namedWindow("Source Image")
    cv.imshow("Source Image", grey_img)
    cv.resizeWindow("Source Image", WINDOW_WIDTH, WINDOW_HEIGHT)

    mean_result = convolution(grey_img, MEAN)
    cv.namedWindow("Mean")
    cv.imshow("Mean", normalise(mean_result))
    cv.resizeWindow("Mean", WINDOW_WIDTH, WINDOW_HEIGHT)

    gaussian_result = convolution(grey_img, GAUSSIAN)
    cv.namedWindow("Gaussian")
    cv.imshow("Gaussian", normalise(gaussian_result))
    cv.resizeWindow("Gaussian", WINDOW_WIDTH, WINDOW_HEIGHT)

    sobelX_grey = convolution(grey_img, SOBEL_X)
    cv.imwrite('sobel-x-grey.jpg', normalise(sobelX_grey))

    sobelY_grey = convolution(grey_img, SOBEL_Y)
    cv.imwrite('sobel-y-grey.jpg', normalise(sobelY_grey))

    sobelX_mean = convolution(mean_result, SOBEL_X)
    cv.imwrite('sobel-x-mean.jpg', normalise(sobelX_mean))

    sobelY_mean = convolution(mean_result, SOBEL_Y)
    cv.imwrite('sobel-y-mean.jpg', normalise(sobelY_mean))

    sobelX_gaussian = convolution(gaussian_result, SOBEL_X)
    cv.imwrite('sobel-x-gaussian.jpg', normalise(sobelX_gaussian))

    sobelY_gaussian = convolution(gaussian_result, SOBEL_Y)
    cv.imwrite('sobel-y-gaussian.jpg', normalise(sobelY_gaussian))

    edge_strength_grey = gradient_magnitude(sobelX_grey, sobelY_grey)
    cv.namedWindow("Edge Strength - Grey")
    cv.imshow("Edge Strength - Grey", normalise(edge_strength_grey))
    cv.resizeWindow("Edge Strength - Grey", WINDOW_WIDTH, WINDOW_HEIGHT)
    cv.imwrite('edge-strength-grey.jpg', normalise(edge_strength_grey))

    edge_strength_mean = gradient_magnitude(sobelX_mean, sobelY_mean)
    cv.namedWindow("Edge Strength - Mean")
    cv.imshow("Edge Strength - Mean", normalise(edge_strength_mean))
    cv.resizeWindow("Edge Strength - Mean", WINDOW_WIDTH, WINDOW_HEIGHT)
    cv.imwrite('edge-strength-mean.jpg', normalise(edge_strength_mean))

    edge_strength_gaussian = gradient_magnitude(sobelX_gaussian, sobelY_gaussian)
    cv.namedWindow("Edge Strength - Gaussian")
    cv.imshow("Edge Strength - Gaussian", normalise(edge_strength_gaussian))
    cv.resizeWindow("Edge Strength - Gaussian", WINDOW_WIDTH, WINDOW_HEIGHT)
    cv.imwrite('edge-strength-gaussian.jpg', normalise(edge_strength_gaussian))

    # Trackbar - Mean
    cv.namedWindow('Threshold - Mean')
    cv.imshow('Threshold - Mean', normalise(edge_strength_mean))
    cv.createTrackbar('Threshold', 'Threshold - Mean', 0, 255, thresholding_mean)
    cv.resizeWindow("Threshold - Mean", WINDOW_WIDTH, WINDOW_HEIGHT)

    # Trackbar - Gaussian
    cv.namedWindow('Threshold - Gaussian')
    cv.imshow('Threshold - Gaussian', normalise(edge_strength_gaussian))
    cv.createTrackbar('Threshold', 'Threshold - Gaussian', 0, 255, thresholding_gaussian)
    cv.resizeWindow("Threshold - Gaussian", WINDOW_WIDTH, WINDOW_HEIGHT)

    # Trackbar - Grey
    cv.namedWindow('Threshold - Grey')
    cv.imshow('Threshold - Grey', normalise(edge_strength_grey))
    cv.createTrackbar('Threshold', 'Threshold - Grey', 0, 255, thresholding_grey)
    cv.resizeWindow("Threshold - Grey", WINDOW_WIDTH, WINDOW_HEIGHT)

    # threshold_test = binary_threshold(normalise(edge_strength_gaussian), 28)
    # cv.namedWindow("t")
    # cv.imshow("t", threshold_test)
    # cv.resizeWindow("t", WINDOW_WIDTH, WINDOW_HEIGHT)

    while True:
        # Press ESC to exit
        if cv.waitKey(1) == 27:
            break

if __name__ == "__main__":
    GAUSSIAN = gaussian(3, 1)
    # GAUSSIAN = gaussian(7, 3)
    MEAN = mean(3)
    main()