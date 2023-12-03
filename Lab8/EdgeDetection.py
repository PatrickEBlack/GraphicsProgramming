#G00410388 - Patrick Black
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Number of Rows and Columns
nrows = 2
ncols = 1

# GrayScale
imgOrig = cv2.imread('ATU.jpg')
imgGray = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY)

# Double Window 1col * 2row
plt.subplot(nrows, ncols, 1), plt.imshow(imgOrig, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols, 2), plt.imshow(imgGray, cmap = 'gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])
plt.show()