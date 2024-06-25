## This code is written by Prof. Vijay Kakani and his team at Computer Vision Laboratory, INHA Univ.##
## Any mpodifications to the current pipeline must be reported to the corresponding authors of the published affliated manuscript.##
## second half of the pre-processed images are subjected to imageJ analysis software, for the processing pipeline: refer to the published article ###
## Permission for code modifications - Contact: vjkakani@inha.ac.kr ##

import cv2
import numpy as np
import matplotlib.pyplot as plt

def resize_and_crop(image_path, output_path, target_size=(512, 512), crop_size=(512,512)):
    # Read the image
    image = cv2.imread(image_path)
    
    # Resize the image
    resized_image = cv2.resize(image, target_size)
    
    # Calculate the crop coordinates
    h, w = resized_image.shape[:2]
    start_x = (w - crop_size[0]) // 2
    start_y = (h - crop_size[1]) // 2
    end_x = start_x + crop_size[0]
    end_y = start_y + crop_size[1]
    
    # Crop the image
    cropped_image = resized_image[start_y:end_y, start_x:end_x]
    
    # Save the cropped image
    cv2.imwrite(output_path, cropped_image)

def plot_image_subplot(image_path, subplot_location, title=''):
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert BGR to RGB for plotting
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Plot the image
    plt.subplot(*subplot_location)
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title(title)

def plot_image_with_canny(image_path, subplot_location, title=''):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Canny edge detection
    edges = cv2.Canny(image, 100, 200)
    
    # Plot the image with edges
    plt.subplot(*subplot_location)
    plt.imshow(edges, cmap='gray')
    plt.axis('off')
    plt.title(title)

def plot_surface_morphology(image_path, subplot_location, title=''):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Plot the image with LUT colormap
    plt.subplot(*subplot_location)
    plt.imshow(image, cmap='jet')
    plt.axis('off')
    plt.title(title)
    
    # Calculate and print porosity value
    ra, rq = calculate_porosity(image_path)
    plt.text(0, 0, f"Ra: {ra:.3f}\nRq: {rq:.3f}", color='white', fontsize=10,
             verticalalignment='top', horizontalalignment='left', backgroundcolor='black')

def calculate_porosity(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Calculate surface height matrix
    surface_height = image.astype(np.float64)

    # Calculate average roughness (Ra)
    ra = np.mean(np.abs(surface_height - np.mean(surface_height)))

    # Calculate root mean square roughness (Rq)
    rq = np.sqrt(np.mean((surface_height - np.mean(surface_height))**2))

    return ra, rq

# Image paths
image_paths = [
    "GEFe@Mn2O3.png",
    "GEMn2O3.png"
]

# Output cropped image paths
crop_paths = [
    "GEFe@Mn2O3_512.png",
    "GEMn2O3_512.png"
]

# Resize and crop the images
for i in range(len(image_paths)):
    resize_and_crop(image_paths[i], crop_paths[i])

# Create a canvas of subplots
fig, axs = plt.subplots(4, 3, figsize=(4, 5))

# Plot the cropped images
for i in range(len(crop_paths)):
    plot_image_subplot(crop_paths[i], (4, 3, i*3+1))

    # Plot Canny edge detection
    plot_image_with_canny(crop_paths[i], (4, 3, i*3+2))

    # Plot surface morphology with porosity value
    plot_surface_morphology(crop_paths[i], (4, 3, i*3+3))

# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()
