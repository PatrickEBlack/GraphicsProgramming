#G00410388 - Patrick Black
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Number of Rows and Columns
nrows = 2
ncols = 2

# GrayScale
imgOrig = cv2.imread('ATU.jpg')
imgGray = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY)

img3by3 = cv2.GaussianBlur(imgGray,(3, 3), 0)
img13by13 = cv2.GaussianBlur(imgGray,(13, 13), 0)

# Double Window 1col * 2row
plt.subplot(nrows, ncols, 1), plt.imshow(cv2.cvtColor(imgOrig,cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols, 2), plt.imshow(imgGray, cmap = 'gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols, 3), plt.imshow(img3by3, cmap = 'gray')
plt.title('3 x 3'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols, 4), plt.imshow(img13by13, cmap = 'gray')
plt.title('13 x 13'), plt.xticks([]), plt.yticks([])
plt.show()