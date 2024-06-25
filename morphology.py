## This code is written by Prof. Vijay Kakani and his team at Computer Vision Laboratory, INHA Univ.##
## Any mpodifications to the current pipeline must be reported to the corresponding authors of the published affliated manuscript.##
## second half of the pre-processed images are subjected to imageJ analysis software, for the processing pipeline: refer to the published article ###
## Permission for code modifications - Contact: vjkakani@inha.ac.kr ##

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to apply surface morphology
def apply_surface_morphology(img):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

# Read the images
img1 = cv2.imread('GEMn2O3.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('GEFe@Mn2O3.png', cv2.IMREAD_GRAYSCALE)

# Apply thresholding
_, thresh1 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)
_, thresh2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)

# Apply surface morphology
morphology1 = apply_surface_morphology(thresh1)
morphology2 = apply_surface_morphology(thresh2)

# Create subplots
plt.subplot(231), plt.imshow(img1, cmap='gray'), plt.title('Original 1')
plt.subplot(232), plt.imshow(thresh1, cmap='gray'), plt.title('Threshold 1')
plt.subplot(233), plt.imshow(morphology1, cmap='gray'), plt.title('Surface Morphology 1')

plt.subplot(234), plt.imshow(img2, cmap='gray'), plt.title('Original 2')
plt.subplot(235), plt.imshow(thresh2, cmap='gray'), plt.title('Threshold 2')
plt.subplot(236), plt.imshow(morphology2, cmap='gray'), plt.title('Surface Morphology 2')

# Show the subplots
plt.show()
