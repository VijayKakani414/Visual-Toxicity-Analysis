## This code is written by Prof. Vijay Kakani and his team at Computer Vision Laboratory, INHA Univ.##
## Any mpodifications to the current pipeline must be reported to the corresponding authors of the published affliated manuscript.##
## Permission for code modifications - Contact: vjkakani@inha.ac.kr ##

import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_roughness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # Calculate surface height matrix
    surface_height = gray.astype(np.float64)

    # Calculate average roughness (Ra)
    ra = np.mean(np.abs(surface_height - np.mean(surface_height)))

    # Calculate root mean square roughness (Rq)
    rq = np.sqrt(np.mean((surface_height - np.mean(surface_height))**2))

    return ra, rq

# Load input images
image_files = ['GEMn2O3.png', 'GEFe@Mn2O3.png']

# Create subplot canvas
fig, axs = plt.subplots(len(image_files), 6, figsize=(12, 7))

# Iterate over images and different sizes
for i, image_file in enumerate(image_files):
    image = cv2.imread(image_file)

    # Plot original image with calculated values
    axs[i, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ra, rq = calculate_roughness(image)
    axs[i, 0].set_title(f'Original\nRa: {ra:.2f}\nRq: {rq:.2f}')
    axs[i, 0].axis('off')

    sizes = [512, 256, 128, 64, 32]
    for j, size in enumerate(sizes):
        resized_image = cv2.resize(image, (size, size))

        # Plot resized image with calculated values
        axs[i, j+1].imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
        ra, rq = calculate_roughness(resized_image)
        axs[i, j+1].set_title(f'Size {size}x{size}\nRa: {ra:.2f}\nRq: {rq:.2f}')
        axs[i, j+1].axis('off')

# Adjust subplot spacing
plt.tight_layout()

# Display the subplot canvas
plt.show()
