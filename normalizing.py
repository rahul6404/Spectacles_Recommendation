import pandas as pd

# Load your CSV data into DataFrame
df = pd.read_csv('insightface_387.csv')
X = df.drop('face_shape', axis=1)
y = df['face_shape']


# Divide each value in each column by the sum of that column
normalized_df = X.apply(lambda x: round(x / x.max(), 2))
normalized_df['face_shape'] = y

# Save the normalized DataFrame to a new CSV file
normalized_df.to_csv('insightface_387_normalized.csv', index=False)

print("Normalized DataFrame has been saved to 'normalized_file.csv'.")


