import cv2
import numpy as np
from matplotlib import pyplot as plt

# Number of rows and columns
nrows = 2
ncols = 3

imgOrig = cv2.imread('ATU1.jpg') # Original Image
# imgOrig = cv2.imread('cartman.jpg') # Cartman Version
imgGray = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY) # Grayscale Image
imgHarris = imgGray.copy() # Copy of GrayScale ATU1.jpg for Harris Corners
imgShiTomasi = imgGray.copy() # Copy of GrayScale ATU1.jpg for Shi Tomasi Corners
imgORB = imgGray.copy() # Copy of GrayScale ATU1.jpg for OBB Key Points Detection

# Apply cv2.cornerHarris() function
harris_corners = cv2.cornerHarris(imgHarris, 2, 3, 0.04)

# Dilate corners for better marking
harris_corners = cv2.dilate(harris_corners, None)

# Define a threshold for extracting large corners
threshold = 0.01 * harris_corners.max()

# Define Corners for ShiTomasi Detection
corners = cv2.goodFeaturesToTrack(imgShiTomasi, 80, 0.01, 10)
corners = np.int64(corners)

#Initiate ORB Detector
orb = cv2.ORB_create()

# Find Key Points with ORB
kp1 = orb.detect(imgORB, None)

# Compute the descriptors with ORB
kp1, des = orb.compute(imgORB, kp1)

# Draw only key points location, not size and orientation
imgORB = cv2.drawKeypoints(imgORB, kp1, None, color=(0, 255, 0), flags = 0)

# Iterate through all the corners and draw them (HarrisCornerDetection)
for i in range(harris_corners.shape[0]):
    for j in range(harris_corners.shape[1]):
        if harris_corners[i, j] > threshold:
            # Draw a red circle at each corner
            cv2.circle(imgHarris, (j, i), 2, (0, 0, 255), -1)

# Iterate through all the corners and draw them (ShiTomasiCornerDetection)
for i in corners:
    x, y = i.ravel()
    cv2.circle(imgShiTomasi, (x, y), 4, 255, -1)

# MatPlotLib Display
plt.subplot(nrows, ncols, 1), plt.imshow(cv2.cvtColor(imgOrig, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols, 2), plt.imshow(imgGray, cmap = 'gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols, 3), plt.imshow(imgHarris, cmap = 'gray')
plt.title('HarrisCorner'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols, 4), plt.imshow(imgShiTomasi, cmap = 'gray')
plt.title('ShiTomasi'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols, 5), plt.imshow(imgORB, cmap = 'gray')
plt.title('ORB'), plt.xticks([]), plt.yticks([])
plt.show() # Display Output Window

