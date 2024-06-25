## This code is written by Prof. Vijay Kakani and his team at Computer Vision Laboratory, INHA Univ.##
## Any mpodifications to the current pipeline must be reported to the corresponding authors of the published affliated manuscript.##
## Permission for code modifications - Contact: vjkakani@inha.ac.kr ##

import matplotlib.pyplot as plt

# Data
categories = ['Original', '512x512', '256x256', '128x128', '64x64', '32x32']
GEMn2O3_data = [37.70, 37.70, 37.49, 37.48, 37.60, 37.22]
GEFeMn2O3_data = [45.63, 45.63, 45.48, 45.46, 45.41, 45.39]

# Set the positions of the x-axis ticks
x_positions = range(len(categories))

# Create the line plots for each category
plt.plot(x_positions, GEMn2O3_data, marker='o', label='GEMn2O3')
plt.plot(x_positions, GEFeMn2O3_data, marker='o', label='GEFe@Mn2O3')

# Set the x-axis labels and title
plt.xlabel('Pixel resolution')
plt.ylabel('Roughness_Ra (%)')
plt.title('Roughness_Ra comparison at various pixel scales')

# Set the x-axis tick labels
plt.xticks(x_positions, categories)

# Add a legend
plt.legend()

# Display the plot
plt.grid()
plt.show()
