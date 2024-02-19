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
GAUSSIAN = np.array([[ 1,  2,  1],
                     [ 2,  4,  2],
                     [ 1,  2,  1]])
MEAN     = np.array([[ 1,  1,  1],
                     [ 1,  1,  1],
                     [ 1,  1,  1],])

def convolution(img: cv.Mat, kernel: np.ndarray):

    result = np.zeros(img.shape)

    kernel_w = kernel.shape[0]
    kernel_h = kernel.shape[1]

    pad = kernel_w // 2
    padded_img = np.pad(img, (pad, pad), mode="constant")

    for col in range(img.shape[1]):
        for row in range(img.shape[0]):
            pixel = padded_img[row:row + kernel_w, col:col + kernel_h]
            result[row, col] = np.sum(pixel * kernel)
    
    # Normalise values to range [0, 255]
    normalised_img = normalise(result)
    
    if ((kernel == SOBEL_X).all()):
        cv.namedWindow("Sobel X")
        cv.imshow("Sobel X", normalised_img)
        cv.resizeWindow("Sobel X", WINDOW_WIDTH, WINDOW_HEIGHT)
    elif ((kernel == SOBEL_Y).all()):
        cv.namedWindow("Sobel Y")
        cv.imshow("Sobel Y", normalised_img)
        cv.resizeWindow("Sobel Y", WINDOW_WIDTH, WINDOW_HEIGHT)
    elif ((kernel == GAUSSIAN).all()):
        cv.namedWindow("Gaussian")
        cv.imshow("Gaussian", normalised_img)
        cv.resizeWindow("Gaussian", WINDOW_WIDTH, WINDOW_HEIGHT)
    elif ((kernel == MEAN).all()):
        cv.namedWindow("Mean")
        cv.imshow("Mean", normalised_img)
        cv.resizeWindow("Mean", WINDOW_WIDTH, WINDOW_HEIGHT)

    return result

def thresholding(img: cv.Mat, threshold: int):
    return np.where(img > threshold, 255, 0)

def normalise(img: cv.Mat):
    result = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255)
    return result.astype(np.uint8)

def main():
    m_img = cv.imread(INPUT_IMAGE)

    if m_img is None:
        print("ERROR: Failed to open image")
        sys.exit()

    grey_img = cv.cvtColor(m_img, cv.COLOR_BGR2GRAY)
    cv.namedWindow("Source Image")
    cv.imshow("Source Image", grey_img)
    cv.resizeWindow("Source Image", WINDOW_WIDTH, WINDOW_HEIGHT)

    convolution(grey_img, SOBEL_X)
    convolution(grey_img, SOBEL_Y)
    convolution(grey_img, GAUSSIAN)
    convolution(grey_img, MEAN)

    while True:
        # Press ESC to exit
        if cv.waitKey(1) == 27:
            break

if __name__ == "__main__":
    main()