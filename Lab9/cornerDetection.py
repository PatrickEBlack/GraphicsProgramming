import cv2
import numpy as np
from matplotlib import pyplot as plt

# Number of rows and columns
nrows = 3
ncols = 1

imgOrig = cv2.imread('ATU1.jpg') # Original Image
imgGray = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY) # Grayscale Image
imgHarris = imgGray.copy() # Copy of Original ATU1.jpg
dst = cv2.cornerHarris(imgGray, 2, 3, 0.1)

# Corner Detection
dst = cv2.dilate(dst, None)
imgOrig[dst>0.01 * dst.max()]=[0, 0, 255]
cv2.imshow('dst', imgOrig)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

# threshold = 0.5; #number between 0 and 1
# for i in range(len(dst)):
#     for j in range(len(dst[i])):
#         if dst[i][j] > (threshold*dst.max()):
#             cv2.circle(imgHarris,(j,i),3,(255, 0, 0),-1)

# cv2.imshow('Harris', imgHarris)

# plt.subplot(nrows, ncols, 1), plt.imshow(cv2.cvtColor(imgOrig, cv2.COLOR_BGR2RGB), cmap = 'gray') # Original
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(nrows, ncols, 2), plt.imshow(imgGray, cmap = 'gray') # Gray Scale
# plt.title('Gray Scale'), plt.xticks([]), plt.yticks([])
#plt.show() # Displays output Window

#cv2.circle(imgHarris, (100, 100), 30, (255, 0, 0), -1) # Copied Image With Blue Cirlce