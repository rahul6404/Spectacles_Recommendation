import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_csv('Merged_Heart_&_Oval.csv')

# Separate features and target
X = df.drop('face_shape', axis=1).astype('category')
y = df['face_shape']

# Encode the labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

# Initialize the XGBClassifier with initial parameters
xgb_model = XGBClassifier(learning_rate=0.05, n_estimators=200, max_depth=10,
                          min_child_weight=1, gamma=0, subsample=0.8,
                          colsample_bytree=0.6, objective='multi:softmax',
                          num_class=4, nthread=4, seed=27, enable_categorical=True)

# Define the grid of hyperparameters to search for XGBClassifier
param_grid_xgb = {
    'max_depth': [3, 4, 5, 6, 7],
    'min_child_weight': [1, 5, 10, 15],
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300]
}

# Set up the k-fold cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize GridSearchCV for XGBClassifier
random_search_xgb = RandomizedSearchCV(xgb_model, param_grid_xgb, scoring='accuracy', n_jobs=-1, cv=kfold)

# Fit the grid search to the data
random_search_xgb.fit(X_train, y_train)

# Print the best parameters and highest accuracy for XGBClassifier
print(f"Best parameters found for XGBClassifier: {random_search_xgb.best_params_}")
print(f"Best accuracy found for XGBClassifier: {random_search_xgb.best_score_}")

# Initialize the RandomForestClassifier
rf_model = RandomForestClassifier()

# Define the grid of hyperparameters to search for RandomForestClassifier
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize GridSearchCV for RandomForestClassifier
random_search_rf = RandomizedSearchCV(rf_model, param_grid_rf, scoring='accuracy', n_jobs=-1, cv=kfold)

# Fit the grid search to the data
random_search_rf.fit(X_train, y_train)

# Print the best parameters and highest accuracy for RandomForestClassifier
print(f"Best parameters found for RandomForestClassifier: {random_search_rf.best_params_}")
print(f"Best accuracy found for RandomForestClassifier: {random_search_rf.best_score_}")

# Define base models for stacking
base_models = [
    ('xgb', random_search_xgb.best_estimator_),
    ('rf', random_search_rf.best_estimator_)
]

# Define meta-model
meta_model = LogisticRegression()

# Initialize StackingClassifier
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model)

# Fit the stacking model
stacking_model.fit(X_train, y_train)

# Make predictions with the stacking model
y_pred_stack = stacking_model.predict(X_test)

# Evaluate the stacking model
accuracy_stack = accuracy_score(y_test, y_pred_stack)
print(f"Improved Model Accuracy with Stacking: {accuracy_stack}")

# Get feature importances from the best XGB model
feature_importances = random_search_xgb.best_estimator_.feature_importances_

# Sort the feature importances in descending order and get the indices
sorted_indices = np.argsort(feature_importances)[::-1]

# Create labels for the horizontal bar chart
labels = X.columns[sorted_indices]

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.barh(range(X.shape[1]), feature_importances[sorted_indices], align='center')
plt.yticks(range(X.shape[1]), labels)
plt.gca().invert_yaxis()
plt.xlabel('Relative Importance')
plt.show()
