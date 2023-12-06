import cv2
import numpy as np
from matplotlib import pyplot as plt

# Declare Number of Rows and Columns
nrows = 2
ncols = 2

# Images
img = cv2.imread('cartman.jpg') # Original
img_new = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
img_1 = cv2.cvtColor(img_new, cv2.COLOR_RGB2HSV) # Convert to HSV

# Split the Channels
(h_channel, s_channel, v_channel) = cv2.split(img_1)

# Display the Images
plt.subplot(nrows, ncols, 1), plt.imshow(img_new, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols, 2), plt.imshow(h_channel, cmap = 'gray')
plt.title('H_Channel'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols, 3), plt.imshow(s_channel, cmap = 'gray')
plt.title('S_Channel'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols, 4), plt.imshow(v_channel, cmap = 'gray')
plt.title('V_Channel'), plt.xticks([]), plt.yticks([])
plt.show()