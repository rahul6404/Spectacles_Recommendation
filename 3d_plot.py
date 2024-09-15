import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

df = pd.read_csv('/merged_file_5.csv')

# Create a figure for the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define a color for each class name
class_colors = {
    'Heart_face_shape': 'r',
    'Oblong_face_shape': 'g',
    'Oval_face_shape': 'b',
    'Round_face_shape': 'y',
    'Square_face_shape': 'c'
}

# Iterate over the class names and plot each class
for class_name, color in class_colors.items():
    # Filter the data for each class
    class_subset = df[df['face_shape'] == class_name]

    # Scatter plot for each class
    ax.scatter(class_subset['r1'], class_subset['r2'], class_subset['r4'], c=color, label=class_name)

# Labels for the axes
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')

# Legend
ax.legend()

# Show plot
plt.show()

