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
MEAN     = np.array([[ 1,  1,  1],
                     [ 1,  1,  1],
                     [ 1,  1,  1],])

def convolution(img: cv.Mat, kernel: np.ndarray):
    if ((kernel == MEAN).all()):
        kernel = (1 / kernel.shape[0]) * MEAN

    result = np.zeros(img.shape)

    kernel_w = kernel.shape[0]
    kernel_h = kernel.shape[1]

    pad = kernel_w // 2
    padded_img = padding(img, pad, pad)

    for col in range(img.shape[1]):
        for row in range(img.shape[0]):
            pixel = padded_img[row:row + kernel_w, col:col + kernel_h]
            result[row, col] = np.sum(pixel * kernel)

    return result

def gaussian(size, sigma=1):
    # Generate numpy array of indices 
    # and subtract each element by the center value
    x, y = np.indices((size, size)) - size // 2

    # G(x, y) = e^-((x^2+y^2)/(2*sigma^2))
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    return kernel

def padding(img: cv.Mat, pad_w: int, pad_h: int):
    w, h = img.shape

    # Initialize result (padded image) with zeros
    padded_img = np.zeros((w + 2 * pad_w, h + 2 * pad_h), dtype=img.dtype)
    padded_img[pad_w:pad_w + w, pad_h:pad_h + h] = img

    return padded_img

# def thresholding(img: cv.Mat, threshold: int):
#     return np.where(img > threshold, 255, 0)

def thresholding_mean(threshold: int):
    _, result = cv.threshold(normalise(edge_strength_mean), threshold, 255, cv.THRESH_BINARY)
    cv.imshow("Threshold - Mean", result)

def thresholding_gaussian(threshold: int):
    _, result = cv.threshold(normalise(edge_strength_gaussian), threshold, 255, cv.THRESH_BINARY)
    cv.imshow("Threshold - Gaussian", result)

def normalise(img: cv.Mat):
    # Normalise values to range [0, 255]
    result = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255)
    return result.astype(np.uint8)

def gradient_magnitude(sobelX: np.ndarray, sobelY: np.ndarray):
    return np.sqrt(sobelX ** 2 + sobelY ** 2)

def main():
    global edge_strength_mean
    global edge_strength_gaussian
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

    sobelX_mean = convolution(mean_result, SOBEL_X)
    cv.namedWindow("Sobel X - Mean")
    cv.imshow("Sobel X - Mean", normalise(sobelX_mean))
    cv.resizeWindow("Sobel X - Mean", WINDOW_WIDTH, WINDOW_HEIGHT)

    sobelY_mean = convolution(mean_result, SOBEL_Y)
    cv.namedWindow("Sobel Y - Mean")
    cv.imshow("Sobel Y - Mean", normalise(sobelY_mean))
    cv.resizeWindow("Sobel Y - Mean", WINDOW_WIDTH, WINDOW_HEIGHT)

    sobelX_gaussian = convolution(gaussian_result, SOBEL_X)
    cv.namedWindow("Sobel X - Gaussian")
    cv.imshow("Sobel X - Gaussian", normalise(sobelX_gaussian))
    cv.resizeWindow("Sobel X - Gaussian", WINDOW_WIDTH, WINDOW_HEIGHT)

    sobelY_gaussian = convolution(gaussian_result, SOBEL_Y)
    cv.namedWindow("Sobel Y - Gaussian")
    cv.imshow("Sobel Y - Gaussian", normalise(sobelY_gaussian))
    cv.resizeWindow("Sobel Y - Gaussian", WINDOW_WIDTH, WINDOW_HEIGHT)

    edge_strength_mean = gradient_magnitude(sobelX_mean, sobelY_mean)
    cv.namedWindow("Edge Strength - Mean")
    cv.imshow("Edge Strength - Mean", normalise(edge_strength_mean))
    cv.resizeWindow("Edge Strength - Mean", WINDOW_WIDTH, WINDOW_HEIGHT)

    edge_strength_gaussian = gradient_magnitude(sobelX_gaussian, sobelY_gaussian)
    cv.namedWindow("Edge Strength - Gaussian")
    cv.imshow("Edge Strength - Gaussian", normalise(edge_strength_gaussian))
    cv.resizeWindow("Edge Strength - Gaussian", WINDOW_WIDTH, WINDOW_HEIGHT)

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

    while True:
        # Press ESC to exit
        if cv.waitKey(1) == 27:
            break

if __name__ == "__main__":
    GAUSSIAN = gaussian(3, 1)
    main()