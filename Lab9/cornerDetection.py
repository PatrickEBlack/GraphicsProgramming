import numpy
import cv2
from matplotlib import pyplot as plt 

# Number Of Columns and Rows
nrows = 2 
ncols = 1

imgOrig = cv2.imread('ATU1.jpg') # Original Image
imgGray = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY) # Grayscale Image

plt.subplot(nrows, ncols, 1), plt.imshow(cv2.cvtColor(imgOrig,cv2.COLOR_BGR2RGB), cmap = 'gray') # Original
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols, 2), plt.imshow(imgGray, cmap = 'gray') # GrayScale
plt.title('Grayscale'), plt.xticks([]), plt.yticks([])

plt.show()