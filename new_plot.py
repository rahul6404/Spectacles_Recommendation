import matplotlib.pyplot as plt
import numpy as np
import pandas as pd# Load your dataset (replace 'your_dataset.csv' with your actual file path)
# Assuming you have features X and labels y
# X should be a 2D array with shape (n_samples, 2)
# y should be an array of class labels (0, 1, 2, 3, 4, etc.)

# Example: Load the Iris dataset
df = pd.read_csv('normalized_file.csv')
X = df[['ratio_2', 'ratio_4']].values
y = df['face_shape'].astype('category').cat.codes.values

# Create a scatter plot
plt.figure(figsize=(8, 6))

# Define colors for each class (you can customize these)
colors = ["red", "blue", "green", "orange", "purple"]

# Scatter plot for each class
for class_label in np.unique(y):
    mask = (y == class_label)
    plt.scatter(X[mask, 0], X[mask, 1], label=f"Class {class_label}", color=colors[class_label])

plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.title("Bivariate 2D Scatter Plot with 5 Classes")
plt.legend()
plt.grid(True)
plt.show()