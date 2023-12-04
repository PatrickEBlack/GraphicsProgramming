import cv2
import numpy as np
from matplotlib import pyplot as plt

# Number of rows and columns
nrows = 3
ncols = 1

imgOrig = cv2.imread('ATU1.jpg') # Original Image
imgGray = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY) # Grayscale Image

imageCopy = imgOrig.copy() # Copy of Original ATU1.jpg
cv2.circle(imageCopy, (100, 100), 30, (255, 0, 0), -1)

plt.subplot(nrows, ncols, 1), plt.imshow(cv2.cvtColor(imgOrig, cv2.COLOR_BGR2RGB), cmap = 'gray') # Original
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols, 2), plt.imshow(imgGray, cmap = 'gray') # Gray Scale
plt.title('Gray Scale'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols, 3), plt.imshow(cv2.cvtColor(imageCopy, cv2.COLOR_BGR2RGB), cmap = 'gray') # Gray Scale
plt.title('Copied Image'), plt.xticks([]), plt.yticks([])

plt.show() # Displays output Window