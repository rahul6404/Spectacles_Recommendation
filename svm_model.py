from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('merged_file_4.csv')

# Separate features and target label
X = df.drop('face_shape', axis=1)
y = df['face_shape']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter range
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}

# Create a GridSearchCV object
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)

# Fit the model for grid search
grid.fit(X_train, y_train)

# Print best parameter after tuning
print(grid.best_params_)

# Print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)

# Predict the labels for the test set
grid_predictions = grid.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, grid_predictions)
print(f'Accuracy of the tuned SVM model: {accuracy:.2f}')

# Print classification report
print(classification_report(y_test, grid_predictions))

