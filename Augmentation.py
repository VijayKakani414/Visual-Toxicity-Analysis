## This code is written by Prof. Vijay Kakani and his team at Computer Vision Laboratory, INHA Univ.##
## Any mpodifications to the current pipeline must be reported to the corresponding authors of the published affliated manuscript.##
## second half of the pre-processed images are subjected to Surface morphology and Roughness analysis via openCV and imageJ analysis software, for the processing pipeline: refer to the published article ###
## Permission for code modifications - Contact: vjkakani@inha.ac.kr ##

import cv2
import os
import numpy as np

# Load the SEM image
img = cv2.imread('GEFe@Mn2O3.png', cv2.IMREAD_GRAYSCALE)

# Define the output directory for augmented images
output_dir = 'AI_variants'

# Create the output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the number of augmented images to generate
num_augmentations = 25

# Define the shape of the output images
output_shape = (512, 512)

# Loop through the number of augmentations and apply random transformations to the SEM image
for i in range(num_augmentations):
    # Randomly select a combination of transformations to apply
    rotation_angle = np.random.randint(0, 360)
    flip_horizontal = np.random.choice([True, False])
    flip_vertical = np.random.choice([True, False])
    scale_factor = np.random.uniform(0.5, 1.5)

    # Apply the selected transformations to the SEM image
    augmented_img = img.copy()
    augmented_img = cv2.rotate(augmented_img, cv2.ROTATE_90_CLOCKWISE) if rotation_angle == 90 else augmented_img
    augmented_img = cv2.rotate(augmented_img, cv2.ROTATE_180) if rotation_angle == 180 else augmented_img
    augmented_img = cv2.rotate(augmented_img, cv2.ROTATE_90_COUNTERCLOCKWISE) if rotation_angle == 270 else augmented_img
    augmented_img = cv2.flip(augmented_img, 1) if flip_horizontal else augmented_img
    augmented_img = cv2.flip(augmented_img, 0) if flip_vertical else augmented_img
    augmented_img = cv2.resize(augmented_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    augmented_img = cv2.resize(augmented_img, output_shape, interpolation=cv2.INTER_CUBIC)

    # Crop the image to the desired output shape
    height, width = augmented_img.shape[:2]
    x_start = max(0, (width - output_shape[1]) // 2)
    y_start = max(0, (height - output_shape[0]) // 2)
    x_end = min(width, x_start + output_shape[1])
    y_end = min(height, y_start + output_shape[0])
    augmented_img = augmented_img[y_start:y_end, x_start:x_end]

    # Save the augmented image to the output directory
    filename = os.path.join(output_dir, f'GEFe@Mn2O3_augmented_{i}.png')
    cv2.imwrite(filename, augmented_img)
