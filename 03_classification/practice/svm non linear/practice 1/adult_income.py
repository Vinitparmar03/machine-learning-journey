# --- Importing required libraries ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

# --- Load dataset ---
dataset = pd.read_csv('adult.csv')

# --- Replace '?' with NaN ---
dataset.replace('?', np.nan, inplace=True)

# --- Fill missing values in specified columns with mode ---
for col in ['workclass', 'occupation', 'native.country']:
    dataset[col].fillna(dataset[col].mode()[0], inplace=True)

# If you want to drop rows with NaN values instead of filling:
# dataset.dropna(subset=['workclass', 'occupation', 'native.country'], inplace=True)

# --- Separate features and target ---
X = dataset.drop('income', axis=1)
y = dataset['income']
original_columns = X.columns

# --- Define numeric and categorical feature columns ---
numeric_cols = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
categorical_cols = ['workclass', 'education', 'marital.status', 'occupation',
                    'relationship', 'race', 'sex', 'native.country']

# --- Create ColumnTransformer for preprocessing ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),  # Scale numerical features
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols)  # One-hot encode categorical features
    ],
    remainder='passthrough'  # Leave other columns (if any) unchanged
)

# --- Split into training and test sets ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# --- Convert X_train and X_test to DataFrame to preserve column order ---
X_train = pd.DataFrame(X_train, columns=original_columns)
X_test = pd.DataFrame(X_test, columns=original_columns)

# --- Apply preprocessing ---
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# --- Encode target labels ---
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# --- Train SVM model with RBF kernel ---
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train_processed, y_train_encoded)

# --- Predict income for a new observation ---
new_observation = pd.DataFrame({
    'age': [82],
    'workclass': ['Private'],
    'fnlwgt': [132870],
    'education': ['HS-grad'],
    'education.num': [9],
    'marital.status': ['widowed'],
    'occupation': ['Exec-managerial'],
    'relationship': ['Not-in-family'],
    'race': ['White'],
    'sex': ['Female'],
    'capital.gain': [0],
    'capital.loss': [4356],
    'hours.per.week': [18],
    'native.country': ['United-States']
}, columns=original_columns)

# --- Preprocess the new observation ---
transformed_new_observation = preprocessor.transform(new_observation)

# --- Make prediction and decode label ---
predicted_income_encoded = classifier.predict(transformed_new_observation)
predicted_income_label = le.inverse_transform(predicted_income_encoded)

print(f"\nPredicted income for the new observation: {predicted_income_label[0]}")

# --- Predict test set results ---
y_pred = classifier.predict(X_test_processed)

# --- Confusion Matrix ---
cm = confusion_matrix(y_test_encoded, y_pred)
print("\nConfusion Matrix:")
print(cm)
print(f"True Negatives (TN): {cm[0, 0]}")
print(f"False Positives (FP): {cm[0, 1]}")
print(f"False Negatives (FN): {cm[1, 0]}")
print(f"True Positives (TP): {cm[1, 1]}")

# --- Accuracy Score ---
accuracy = accuracy_score(y_test_encoded, y_pred)
print(f"\nAccuracy Score: {accuracy:.4f}")

# --- Show Predicted vs Actual values ---
y_pred = np.array(y_pred)
y_test_encoded = np.array(y_test_encoded)

comparison_array = np.concatenate(
    (y_pred.reshape(-1, 1), y_test_encoded.reshape(-1, 1)), axis=1
)

print("Predicted vs Actual values (on test set):")
print(comparison_array)
