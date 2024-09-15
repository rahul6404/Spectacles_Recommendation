import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_csv('insightface_387.csv')

# Separate features and target
X = df.drop('face_shape', axis=1).astype('category')
y = df['face_shape']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
# print(y_test)
# print(y_train)

# StandardScaling features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Encode the labels to integers
le1 = LabelEncoder()
y_train = le1.fit_transform(y_train)
y_test = le1.fit_transform(y_test)
mapping = dict(zip(le1.classes_, le1.transform(le1.classes_)))
print(mapping)


# Initialize the XGBClassifier with initial parameters
model = XGBClassifier(learning_rate=0.05, n_estimators=300, max_depth=3,
                      min_child_weight=1, gamma=0.5, subsample=0.6,
                      colsample_bytree=0.6, objective='multi:softmax',
                      num_class=5, nthread=-1, seed=27, enable_categorical=True)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# model = RandomForestClassifier()

# model = SVC(kernel="rbf", degree=3, gamma='scale')
# svc_params = {
#     'kernel':['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
#     'gamma':['scale', 'auto']
# }


# Define the grid of hyperparameters to search
# param_grid = {
#     'max_depth': [3, 4, 5],
#     'min_child_weight': [1, 5, 10],
#     'gamma': [0.5, 1, 1.5, 2],
#     'subsample': [0.6, 0.8, 1.0],
#     'colsample_bytree': [0.6, 0.8, 1.0],
#     'n_estimators': [100, 200, 300]
# }

# rf_params = {
#     "max_depth":[5, 8, 15, None, 10],
#     "max_features":[5, 10, 15, 21],
#     "min_samples_split":[2, 8, 15, 20],
#     "n_estimators": [100, 200, 500, 1000]
# }

# param_grid = {
#     'max_depth': [3, 4, 5, 6, 7],
#     'min_child_weight': [1, 5, 10, 15],
#     'gamma': [0.5, 1, 1.5, 2, 5],
#     'subsample': [0.6, 0.8, 1.0],
#     'colsample_bytree': [0.6, 0.8, 1.0],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'n_estimators': [100, 200, 300]
# }


# Set up the k-fold cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize GridSearchCV
# grid_search = GridSearchCV(model, svc_params, scoring='accuracy', n_jobs=-1, cv=kfold)
# grid_search.fit(X_train, y_train)

# RandomizedSearchCV Initialization
# random_search = RandomizedSearchCV(model, param_grid, scoring='accuracy', n_jobs=-1, cv=kfold)
# random_search.fit(X_train, y_train)

# Print the best parameters and highest accuracy
# print(f"Best parameters found: {grid_search.best_params_}")
# print(f"Best accuracy found: {grid_search.best_score_}")

# Use the best estimator to make predictions
# best_model = grid_search.best_estimator_
# y_pred = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Improved Model Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# # Check for underfitting or overfitting
# training_accuracy = accuracy_score(y_train, best_model.predict(X_train))
# validation_accuracy = grid_search.best_score_
# #
# print(f"Training Accuracy: {training_accuracy}")
# print(f"Validation Accuracy: {validation_accuracy}")
#
# if training_accuracy > validation_accuracy:
#     if (training_accuracy - validation_accuracy) > 0.1: # arbitrary threshold
#         print("The model may be overfitting.")
#     else:
#         print("The model is probably fitting well.")
# else:
#     if validation_accuracy - training_accuracy > 0.1: # arbitrary threshold
#         print("The model may be underfitting.")
#     else:
#         print("The model is probably fitting well.")
#
# # Save the model for future use
# best_model.save_model('face_shape_classifier.json')


# Fit the model on the training data
# best_model.fit(X_train, y_train)

# Get feature importances
feature_importances = model.feature_importances_

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

# import pickle
# with open('model_2.pkl', 'wb') as file:
#     pickle.dump(model, file)
# with open('scaler_2.pkl', 'wb') as file:
#     pickle.dump(scaler, file)




