import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load your CSV data into a Pandas DataFrame (replace 'merged_file_4.csv' with your actual file path)
df = pd.read_csv('merged_file_3.csv')

# Assuming your features are named 'norm_chin_angle_1' and 'norm_chin_angle_2'
features = df[['chin_angle_1', 'chin_angle_2']]

# Choose the number of clusters (in your case, 5)
num_clusters = 4
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(features)

# Add cluster labels to your DataFrame
df['face_shape'] = kmeans.labels_

# Create a dictionary mapping cluster numbers to face shape labels
cluster_to_face_shape = {0: 'Heart', 1: 'Square', 2: 'Oblong', 3: 'Round'}

# Visualize the clusters with face shape labels
plt.figure(figsize=(8, 6))
for cluster_num, face_shape in cluster_to_face_shape.items():
    cluster_points = df[df['face_shape'] == cluster_num]
    plt.scatter(cluster_points['chin_angle_1'], cluster_points['chin_angle_2'], label=face_shape)

plt.xlabel('chin_angle_1')
plt.ylabel('chin_angle_2')
plt.title('K-Means Clustering: Feature 1 vs. Feature 2')
plt.legend()
plt.show()
