import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('merged_file_4.csv')

# Separate features and target label
X = df.drop('face_shape', axis=1).values
y = df['face_shape'].values

# Perform t-SNE
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
X_tsne = tsne.fit_transform(X)

# Create a scatter plot
plt.figure(figsize=(12, 8))
for shape in df['face_shape'].unique():
    indices = (y == shape)
    plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], label=shape, alpha=0.7)

plt.legend()
plt.title('t-SNE visualization of face shapes')
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.show()
