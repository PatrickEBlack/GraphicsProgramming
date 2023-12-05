import cv2
import numpy as np
from matplotlib import pyplot as plt

# Declare Number of Rows and Columns
nrows = 1
ncols = 3

# Image 1 - Eric Cartman
img1 = cv2.imread('cartman.jpg')

# Split the Image into channels
(b_channel, g_channel, r_channel) = cv2.split(img1)

# Display The Images
plt.subplot(nrows, ncols, 1), plt.imshow(b_channel, cmap = 'gray')
plt.title('Blue Channel'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols, 2), plt.imshow(g_channel, cmap = 'gray')
plt.title('Green Channel'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols, 3), plt.imshow(r_channel, cmap = 'gray')
plt.title('Red Channel'), plt.xticks([]), plt.yticks([])
plt.show()