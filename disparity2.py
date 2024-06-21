import numpy as np
import cv2
import sys
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt



# ================================================
#
def getDepthMap(imL, imR, numDisparities, blockSize, k):
    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

    disparity = stereo.compute(imL, imR)
    disparity = disparity - disparity.min()
    disparity = disparity.astype(np.float32) / 16.0

    depth = 1 / (disparity + k)

    return depth
# ================================================

# ================================================
#
def segmentation(imL, imR, depth, threshold):
    foreground_mask = (depth < threshold * depth.max()).astype(np.uint8) * 255
    background_mask = 255 - foreground_mask

    foreground = cv2.bitwise_and(imL, imL, mask=foreground_mask)
    blurred_imR = cv2.GaussianBlur(imR, (21, 21), 0)
    background = cv2.bitwise_and(blurred_imR, blurred_imR, mask=background_mask)

    result = cv2.bitwise_or(foreground, cv2.cvtColor(background, cv2.COLOR_GRAY2BGR))

    cv2.namedWindow('Foreground', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Background', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)

    cv2.imshow('Foreground', foreground)
    cv2.imshow('Background', background)
    cv2.imshow('Result', result)

    cv2.imwrite("foreground.png", foreground)
    cv2.imwrite("background.png", background)
    cv2.imwrite("result.png", result)
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

        cv2.setTrackbarPos('numDisparities', 'Depth', numDisp)

    try:    
        depth = getDepthMap(imgL, imgR, numDisp, cv2.getTrackbarPos('blockSize', 'Depth'), cv2.getTrackbarPos('k', 'Depth'))
        depthImg = np.interp(depth, (depth.min(), depth.max()), (0.0, 1.0))
        cv2.imshow('Depth', depthImg)
        segmentation(coloredL, imgR, depth, 0.9)
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

        cv2.setTrackbarPos('blockSize', 'Depth', blockSize)

    try:
        depth = getDepthMap(imgL, imgR, cv2.getTrackbarPos('numDisparities', 'Depth'), blockSize, cv2.getTrackbarPos('k', 'Depth'))
        depthImg = np.interp(depth, (depth.min(), depth.max()), (0.0, 1.0))
        cv2.imshow('Depth', depthImg)
        segmentation(coloredL, imgR, depth, 0.9)
    except:
        print("==============================================")
        print("\nk trackbar not initialized yet\n")
        print("==============================================\n")
# ================================================

# ================================================
#
def kTrack(k):
    depth = getDepthMap(imgL, imgR, cv2.getTrackbarPos('numDisparities', 'Depth'), cv2.getTrackbarPos('blockSize', 'Depth'), k)
    depthImg = np.interp(depth, (depth.min(), depth.max()), (0.0, 1.0))
    cv2.imshow('Depth', depthImg)
    segmentation(coloredL, imgR, depth, 0.9)
# ================================================
#
if __name__ == '__main__':
    # Load left image
    filename = 'girlL.png'
    coloredL = cv2.imread(filename)
    imgL = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    #
    if imgL is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()

    # Load right image
    filename = 'girlR.png'
    coloredR = cv2.imread(filename)
    imgR = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    #
    if imgR is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()

    cv2.namedWindow('Source Left', cv2.WINDOW_NORMAL)
    cv2.imshow('Source Left', imgL)

    # Create a window to display the image in
    cv2.namedWindow('Depth', cv2.WINDOW_NORMAL)

    # Get depth map
    depth = getDepthMap(imgL, imgR, 64, 5, 1)

    # Normalise for display
    depthImg = np.interp(depth, (depth.min(), depth.max()), (0.0, 1.0))

    # Show result
    cv2.imshow('Depth', depthImg)

    # Show 3D plot of the scene
    # plot(disparity)

    cv2.createTrackbar('numDisparities', 'Depth', 64, 255, numDispTrack)
    cv2.createTrackbar('blockSize', 'Depth', 5, 255, blockSizeTrack)
    cv2.createTrackbar('k', 'Depth', 0, 255, kTrack)

    # Wait for spacebar press or escape before closing,
    # otherwise window will close without you seeing it
    while True:
        key = cv2.waitKey(1)
        if key == ord(' ') or key == 27:
            break

    cv2.destroyAllWindows()
