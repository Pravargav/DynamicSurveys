import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the dataset
df = pd.read_csv(r'C:\Users\dell\DynamicSurveys\surveyAPI-Flask\pythonProject\Disease_symptom_and_patient_profile_dataset.csv')

# Separate features and target
X = df.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9]].values
y = df.iloc[:, 0].values

# Encode categorical variables
le = LabelEncoder()
for i in range(X.shape[1]):
    X[:, i] = le.fit_transform(X[:, i])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initialize SVC and define the parameter grid for GridSearchCV
svc = SVC()
param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Kernel types
    'gamma': ['scale', 'auto'],  # Kernel coefficient for 'rbf', 'poly', and 'sigmoid'
}

# Perform GridSearchCV with 10-fold cross-validation
grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best parameters from GridSearchCV
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Train the SVC model with the best parameters
best_svc = SVC(**best_params)
best_svc.fit(X_train, y_train)

# Predict on the test set
y_pred = best_svc.predict(X_test)

# Evaluate the model
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)

print("Confusion Matrix:\n", cm)
print("Accuracy Score:", ac)

# Optional: Perform cross-validation on the entire training set to check consistency
accuracies = cross_val_score(estimator=best_svc, X=X_train, y=y_train, cv=2)
print("Cross-Validation Mean Accuracy:", accuracies.mean())
print("Cross-Validation Standard Deviation:", accuracies.std())

import numpy as np
import joblib

# Predict on the test set using the best SVC model
y_pred_test = best_svc.predict(X_test)

# Predict for a single row of data (first example)
single_row_1 = np.array([[1, 0, 1, 0, 23, 0, 2, 2, 0]])
y_pred_single_1 = best_svc.predict(single_row_1)
print('Prediction for the single row (first example):', y_pred_single_1[0])

# Predict for another single row of data (second example)
single_row_2 = np.array([[1, 1, 1, 0, 154, 1, 2, 3, 1]])

# Save the trained model to a file using joblib
joblib.dump(best_svc, r'C:\Users\dell\surveyApi\pythonProject\modelx\model2x.pkl')

# Load the model from the file
loaded_model = joblib.load(r'C:\Users\dell\surveyApi\pythonProject\modelx\model2x.pkl')

# Predict using the loaded model on the second single row
y_pred_single_2 = loaded_model.predict(single_row_2)
print('Prediction for the single row (second example):', y_pred_single_2[0])
