#G00410388 - Patrick Black
import cv2
import numpy as np
from matplotlib import pyplot as plt

# GrayScale
img = cv2.imread('ATU.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Window', gray_img)
cv2.waitKey()