import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read in Images
img1 = cv2.imread('cartman.jpg', cv2.IMREAD_GRAYSCALE) # Query image
img2 = cv2.imread('cartman1.jpg', cv2.IMREAD_GRAYSCALE) # Train image

# Initiate ORB Detector
orb = cv2.ORB_create()

# Find the Keypoints and descriptions With ORB
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

# Match Descriptors
matches = bf.match(des1, des2)

# Sort them in the order of their distance
matches = sorted(matches, key = lambda x:x.distance)

# Draw the first 10 Matches
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Plot Results
plt.imshow(img3), plt.show()