from scipy.ndimage import maximum_filter
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

import os

WINDOW_WIDTH = 500
WINDOW_HEIGHT = 500

def HarrisPointsDetector(img, threshold, neighbourSize=7, sobelSize=3, k=0.05):
    # Calculate x-gradients, y-gradients and orientation
    Ix = cv.Sobel(img, cv.CV_64F, 1, 0, sobelSize, borderType=cv.BORDER_REFLECT)
    Iy = cv.Sobel(img, cv.CV_64F, 0, 1, sobelSize, borderType=cv.BORDER_REFLECT)
    theta = np.rad2deg(np.arctan2(Iy, Ix))

    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    IxIy = Ix * Iy

    GAUSSIAN_SIZE = (5, 5)
    SIGMA_VAL = 0.5

    # 5x5 Gaussian mask with 0.5 sigma
    S_Ix2 = cv.GaussianBlur(Ix2, GAUSSIAN_SIZE, SIGMA_VAL, borderType=cv.BORDER_REFLECT)
    S_Iy2 = cv.GaussianBlur(Iy2, GAUSSIAN_SIZE, SIGMA_VAL, borderType=cv.BORDER_REFLECT)
    S_IxIy = cv.GaussianBlur(IxIy, GAUSSIAN_SIZE, SIGMA_VAL, borderType=cv.BORDER_REFLECT)

    # Calculate determinant of M, trace of M and R, where k = 0.05
    M_det = (S_Ix2 * S_Iy2) - (S_IxIy ** 2)
    M_trace = S_Ix2 + S_Iy2
    R = M_det - k * M_trace ** 2
    
    # Thresholding
    max_val = R.max()
    R_thresholded = np.where(R > threshold * max_val, R, 0)

    # Non-maximum Suppression
    keypoints = []
    local_maxima = maximum_filter(R_thresholded, (neighbourSize, neighbourSize), mode='reflect')
    R_maxima = np.logical_and(R_thresholded == local_maxima, R_thresholded != 0)
    idx_maxima = np.argwhere(R_maxima)
    for idx in idx_maxima:
        keypoints.append((float(idx[0]), float(idx[1]), theta[idx[0], idx[1]]))

    keypoints = [cv.KeyPoint(y, x, 1, orientation) for x, y, orientation in keypoints]

    return keypoints

def featureDescriptor(img, keypoints, score_type=None, built_in=False):
    if not built_in:
        orb = cv.ORB.create()
        kp, des = orb.compute(img, keypoints)
    else:
        orb = cv.ORB.create(scoreType=score_type)
        kp = orb.detect(img, None)
        kp, des = orb.compute(img, kp)

    return kp, des

def SSDFeatureMatcher(sourceDesc, targetDesc):
    matches = []
    
    for i in range(sourceDesc.shape[0]):
        dist = cdist([sourceDesc[i]], targetDesc, 'sqeuclidean').flatten()
        minIdx = np.argmin(dist, axis=0)
        minDist = dist[minIdx]
        matches.append(cv.DMatch(i, minIdx, minDist))

    return matches

def RatioFeatureMatcher(sourceDesc, targetDesc, threshold):
    matches = []

    for i in range(sourceDesc.shape[0]):
        dist = cdist([sourceDesc[i]], targetDesc, 'sqeuclidean').flatten()
        sorted_dist = dist.copy()
        sorted_dist.sort()

        # Compute ratio of distances if there are at least 2 calculated distances
        if len(sorted_dist) > 1:
            ratio = (sorted_dist[0] / sorted_dist[1])
        else:
            ratio = 0
        
        # Ratio test
        if ratio < threshold:
            minIdx = np.argwhere(dist == sorted_dist[0]).flatten()[0]
            minDist = dist[minIdx]
            matches.append(cv.DMatch(i, minIdx, minDist))

    return matches

def main(ratioTest, scoreType, builtIn, T, ratio_T):
    # CHANGE THIS
    threshold = T
    ratio_threshold = ratio_T
    ratio_test = ratioTest
    score_type = scoreType
    built_in = builtIn
    
    img_dir = "img"
    result_dir = ("result-ratio-" + str(T)) if ratio_test else ("result-ssd-" + str(T))
    
    if score_type == cv.ORB_HARRIS_SCORE and builtIn == True:
        result_dir = "result-ratio-harris" if ratio_test else "result-ssd-harris"
    elif score_type == cv.ORB_FAST_SCORE and builtIn == True:
        result_dir = "result-ratio-fast" if ratio_test else "result-ssd-fast"

    MODE = "SSD + Ratio" if ratio_test else "SSD"
    print("Running program in mode : " + MODE)
    print("Threshold:", threshold, ", Ratio Threshold:", ratio_threshold)

    src = cv.imread("bernieSanders.jpg")
    src_grey = cv.imread("bernieSanders.jpg", cv.IMREAD_GRAYSCALE)
    src_kp = HarrisPointsDetector(src_grey, threshold)
    print("Source - Harris Done!")
    src_kp, src_des = featureDescriptor(src, src_kp, score_type, built_in)

    for filename in os.listdir(img_dir):
        if T == 0:
            print("Feature Matching is not performed as threshold is 0")
            break

        print(filename)
        f = os.path.join(img_dir, filename)

        target = cv.imread(f)
        target_grey = cv.imread(f, cv.IMREAD_GRAYSCALE)
        target_kp = HarrisPointsDetector(target_grey, threshold)
        print("Target - Harris Done!")
        target_kp, target_des = featureDescriptor(target, target_kp, score_type, built_in)
            
        try:
            if ratio_test:
                matches = RatioFeatureMatcher(src_des, target_des, ratio_threshold)
                print("Ratio Test Done!")
            else:
                print("SSD Matching...")
                matches = SSDFeatureMatcher(src_des, target_des)
                print("SDD Done!")
        except:
            print("No keypoints found for", filename, ", Keypoints:", target_des)
            continue

        matches = sorted(matches, key=lambda x:x.distance)

        result = cv.drawMatches(src, src_kp, target, target_kp, matches[:10], None)
        result_out = os.path.join(result_dir, filename)
        cv.imwrite(result_out, result)
    
    print("FINISHED")

    if not built_in:
        return len(src_kp)
    
    return 0

if __name__ == "__main__":
    # thresholds = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    # kpNumCustomHarris = []
    # kpNumORBHarris = []
    # kpNumORBFAST = []

    # for t in thresholds:
    #     kpNumCustomHarris.append(main(False, None, False, t, 0.7))
    #     main(True, None, False, t, 0.7)

    # print("Initiating Built-In Mode...")

    main(False, cv.ORB_HARRIS_SCORE, True, 0.1, 0.7)
    main(True, cv.ORB_HARRIS_SCORE, True, 0.1, 0.7)

    main(False, cv.ORB_FAST_SCORE, True, 0.1, 0.7)
    main(True, cv.ORB_FAST_SCORE, True, 0.1, 0.7)

    # print("Custom Harris Keypoints:", kpNumCustomHarris)
    # print("ORB Harris Keypoints:", kpNumORBHarris)
    # print("ORB FAST Keypoints:", kpNumORBFAST)

    # x = np.array([1, 2.5, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    # y = np.array(kpNumCustomHarris)

    # plt.plot(x, y)
    # plt.title('Custom Harris')
    # plt.ylabel('Number of Keypoints')
    # plt.xlabel('Threshold (% of max R)')
    # for xy in zip(x, y):
    #     plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
    # plt.grid()
    # plt.show()