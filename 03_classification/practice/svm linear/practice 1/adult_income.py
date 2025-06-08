import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

# Load dataset
dataset = pd.read_csv('adult.csv')

# Replace '?' with np.nan to handle missing values
dataset.replace('?', np.nan, inplace=True)

# Fill missing categorical values with the mode of each column
for col in ['workclass', 'occupation', 'native.country']:
    dataset[col].fillna(dataset[col].mode()[0], inplace=True)

# Optional: drop rows with missing values (commented out)
# dataset.dropna(subset=['workclass', 'occupation', 'native.country'], inplace=True)

# Separate features and target variable
X = dataset.drop('income', axis=1)
y = dataset['income']
original_columns = X.columns  # Save original columns for consistent DataFrame usage

# Define numeric and categorical columns
numeric_cols = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
categorical_cols = ['workclass', 'education', 'marital.status', 'occupation',
                    'relationship', 'race', 'sex', 'native.country']

# Setup preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),  # Scale numeric features
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols)  # One-hot encode categoricals
    ],
    remainder='passthrough'  # Leave other columns (if any) unchanged
)

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Convert back to DataFrame to maintain column structure for preprocessing
X_train = pd.DataFrame(X_train, columns=original_columns)
X_test = pd.DataFrame(X_test, columns=original_columns)

# Fit preprocessor on training data and transform both train and test sets
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Encode target labels to integers
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Train Support Vector Classifier with linear kernel
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train_processed, y_train_encoded)

# Create a new sample observation for prediction
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

# Preprocess the new observation
transformed_new_observation = preprocessor.transform(new_observation)

# Predict and decode the income class for the new observation
predicted_income_encoded = classifier.predict(transformed_new_observation)
predicted_income_label = le.inverse_transform(predicted_income_encoded)
print(f"\nPredicted income for the new observation: {predicted_income_label[0]}")

# Predict on test dataset
y_pred = classifier.predict(X_test_processed)

# Generate confusion matrix
cm = confusion_matrix(y_test_encoded, y_pred)
print("\nConfusion Matrix:")
print(cm)
print(f"True Negatives (TN): {cm[0, 0]}")
print(f"False Positives (FP): {cm[0, 1]}")
print(f"False Negatives (FN): {cm[1, 0]}")
print(f"True Positives (TP): {cm[1, 1]}")

# Calculate and print accuracy score
accuracy = accuracy_score(y_test_encoded, y_pred)
print(f"\nAccuracy Score: {accuracy:.4f}")

# For comparison: predicted vs actual labels on test set
comparison_array = np.concatenate(
    (y_pred.reshape(-1, 1), y_test_encoded.reshape(-1, 1)), axis=1
)
print("\nPredicted vs Actual values (on test set):")
print(comparison_array)
