import numpy as np
import cv2
import sys
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt



# ================================================
#
def getDisparityMap(imL, imR, numDisparities, blockSize):
    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

    disparity = stereo.compute(imL, imR)
    disparity = disparity - disparity.min() + 1 # Add 1 so we don't get a zero depth, later
    disparity = disparity.astype(np.float32) / 16.0 # Map is fixed point int with 4 fractional bits

    return disparity # floating point image
# ================================================




# ================================================
#
def plot(disparity):
    # This just plots some sample points.  Change this function to
    # plot the 3D reconstruction from the disparity map and other values
    baseline = 174.019
    f = 5806.559
    doffs = 114.291
    threshold = 0.98
    size = disparity.shape[0] * disparity.shape[1]

    x = np.zeros(size, dtype=float)
    y = np.zeros(size, dtype=float)
    z = np.zeros(size, dtype=float)
    
    i = 0
    for r in range(disparity.shape[0]):
        for c in range(disparity.shape[1]):
                depth = baseline * (f / (disparity[r, c] + doffs))
                x[i] = ((c * depth) / f) - (baseline / 2)
                y[i] = (r * depth) / f
                z[i] = depth
                i += 1

    idx = np.argwhere(np.logical_or(z > threshold * z.max(), z == z.min()))
    z = np.delete(z, idx)
    x = np.delete(x, idx)
    y = np.delete(y, idx)
    
    # Plt depths
    ax = plt.axes(projection='3d')
    ax.scatter(x, z, y, s=1)

    # Labels
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')

    ax.view_init(0, 0)

    plt.savefig('myplot.pdf', bbox_inches='tight') # Can also specify an image, e.g. myplot.png
    plt.show()
# ================================================



# ================================================
#
def numDispTrack(numDisp):
    if numDisp % 16 != 0:
        print("==============================================")
        print("\nnumDisparities not divisible by 16!")
        numDisp = numDisp - (numDisp % 16)
        print("Setting numDisparities to nearest value:", numDisp, "\n")
        print("==============================================\n")

        cv2.setTrackbarPos('numDisparities', 'Disparity', numDisp)

    try:    
        disparity = getDisparityMap(imgL, imgR, numDisp, cv2.getTrackbarPos('blockSize', 'Disparity'))
        disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))
        cv2.imshow('Disparity', disparityImg)
    except:
        print("==============================================")
        print("\nblockSize trackbar not initialized yet\n")
        print("==============================================\n")
# ================================================

# ================================================
#
def blockSizeTrack(blockSize):
    if blockSize % 2 == 0:
        print("==============================================")
        print("\nblockSize is not odd!")
        blockSize += 1
        print("Setting blockSize to nearest odd value:", blockSize, "\n")
        print("==============================================\n")

        cv2.setTrackbarPos('blockSize', 'Disparity', blockSize)

    disparity = getDisparityMap(imgL, imgR, cv2.getTrackbarPos('numDisparities', 'Disparity'), blockSize)
    disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))
    cv2.imshow('Disparity', disparityImg)
# ================================================
#
if __name__ == '__main__':
    # ================================================
    # 1.1 Focal Length Calculation
    focal_length_px = 5806.559
    sensor_width = 22.2
    img_width = 3088
    focal_length_mm = (focal_length_px * sensor_width) / img_width
    print("==============================================")
    print("\nFocal length in mm:", focal_length_mm, "mm\n")
    print("==============================================\n")
    # ================================================

    cannyMin = 80
    cannyMax = 160

    # Load left image
    filename = 'umbrellaL.png'
    imgL = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite("umbrellaL-grey.png", imgL)
    # imgL = cv2.Canny(imgL, cannyMin, cannyMax)
    #
    if imgL is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()

    # Load right image
    filename = 'umbrellaR.png'
    imgR = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    # imgR = cv2.Canny(imgR, cannyMin, cannyMax)
    #
    if imgR is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()

    cv2.namedWindow('Source Left', cv2.WINDOW_NORMAL)
    cv2.imshow('Source Left', imgL)

    # Create a window to display the image in
    cv2.namedWindow('Disparity', cv2.WINDOW_NORMAL)

    # Get disparity map
    disparity = getDisparityMap(imgL, imgR, 64, 5)

    # Normalise for display
    disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))

    # Show result
    cv2.imshow('Disparity', disparityImg)

    # Show 3D plot of the scene
    # plot(disparity)

    cv2.createTrackbar('numDisparities', 'Disparity', 64, 255, numDispTrack)
    cv2.createTrackbar('blockSize', 'Disparity', 5, 255, blockSizeTrack)

    # Wait for spacebar press or escape before closing,
    # otherwise window will close without you seeing it
    while True:
        key = cv2.waitKey(1)
        if key == ord(' ') or key == 27:
            break

    cv2.destroyAllWindows()
