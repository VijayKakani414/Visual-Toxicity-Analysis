## This code is written by Prof. Vijay Kakani and his team at Computer Vision Laboratory, INHA Univ.##
## Any mpodifications to the current pipeline must be reported to the corresponding authors of the published affliated manuscript.##
## second half of the pre-processed images are subjected to imageJ analysis software, for the processing pipeline: refer to the published article ###
## Permission for code modifications - Contact: vjkakani@inha.ac.kr ##

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_surface(image, subplot_location, title=''):
    # Read the image and convert it to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create x, y coordinates
    x, y = np.meshgrid(np.arange(image_gray.shape[1]), np.arange(image_gray.shape[0]))
    
    # Plot the 3D surface morphology using the jet colormap
    ax = plt.subplot(*subplot_location, projection='3d')
    ax.plot_surface(x, y, image_gray, cmap='jet')
    plt.title(title)
    plt.axis('off')

def plot_ridges_and_valleys(image, subplot_location, title=''):
    # Convert the image to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate the Scharr gradients in both x and y directions
    grad_x = cv2.Scharr(image_gray, cv2.CV_64F, 1, 0)
    grad_y = cv2.Scharr(image_gray, cv2.CV_64F, 0, 1)
    
    # Calculate the magnitude of the gradients
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Plot the 2D map of ridges and valleys
    plt.subplot(*subplot_location)
    plt.imshow(magnitude, cmap='jet')
    plt.title(title)
    plt.axis('off')

def plot_canny_edges(image, subplot_location, title=''):
    # Convert the image to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(image_gray, 100, 200)
    
    # Plot the Canny edges
    plt.subplot(*subplot_location)
    plt.imshow(edges, cmap='gray')
    plt.title(title)
    plt.axis('off')

# Load the input images
image_files = ['GEMn2O3.png', 'GEFe@Mn2O3.png']
images = [cv2.imread(file) for file in image_files]

# Create the canvas of subplots
fig, axs = plt.subplots(3, len(image_files), figsize=(12, 8))

# Plot 3D surface morphology, ridges and valleys, and Canny edges for each image
for i in range(len(image_files)):
    plot_canny_edges(images[i], (3, len(image_files), i+1), title='')
    plot_ridges_and_valleys(images[i], (3, len(image_files), i+1+len(image_files)), title='')
    plot_3d_surface(images[i], (3, len(image_files), i+1+2*len(image_files)), title='')

# Adjust subplot spacing
plt.tight_layout()

# Show the plot
plt.show()
