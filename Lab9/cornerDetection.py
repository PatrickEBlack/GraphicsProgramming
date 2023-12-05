import cv2
import numpy as np
from matplotlib import pyplot as plt

# Number of rows and columns
nrows = 3
ncols = 1

imgOrig = cv2.imread('ATU1.jpg') # Original Image
imgGray = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY) # Grayscale Image
imgHarris = imgGray.copy() # Copy of GrayScale ATU1.jpg for Harris Corners
imgShiTomasi = imgGray.copy() # Copy of GrayScale ATU1.jpg for Shi Tomasi Corners

# Normalize to 8-bit
gray_image = np.float32(imgGray)

# Apply cv2.cornerHarris() function
harris_corners = cv2.cornerHarris(imgGray, 2, 3, 0.04)

# Dilate corners for better marking
harris_corners = cv2.dilate(harris_corners, None)

# Define a threshold for extracting large corners
threshold = 0.01 * harris_corners.max()

# Define Corners for ShiTomasi Detection
corners = cv2.goodFeaturesToTrack(imgGray, 40, 0.01, 10)
corners = np.int64(corners)

# Iterate through all the corners and draw them (HarrisCornerDetection)
for i in range(harris_corners.shape[0]):
    for j in range(harris_corners.shape[1]):
        if harris_corners[i, j] > threshold:
            # Draw a red circle at each corner
            cv2.circle(imgOrig, (j, i), 2, (0, 0, 255), -1)

# Iterate through all the corners and draw them (ShiTomasiCornerDetection)
for i in corners:
    x, y = i.ravel()
    cv2.circle(imgOrig, (x, y), 4, 255, -1)

# Display the image with corners
cv2.imshow('Corner Detection', imgOrig)
cv2.waitKey(0)
cv2.destroyAllWindows()

# plt.subplot(nrows, ncols, 1), plt.imshow(cv2.cvtColor(imgOrig, cv2.COLOR_BGR2RGB), cmap = 'gray') # Original
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(nrows, ncols, 2), plt.imshow(imgGray, cmap = 'gray') # Gray Scale
# plt.title('Gray Scale'), plt.xticks([]), plt.yticks([])
#plt.show() # Displays output Window

#cv2.circle(imgHarris, (100, 100), 30, (255, 0, 0), -1) # Copied Image With Blue Cirlce