# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Preprocessing and modeling libraries
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.compose import ColumnTransformer

# Load the dataset
dataset = pd.read_csv('bank.csv')

# Replace '?' with NaN
dataset.replace('?', np.nan, inplace=True)

# Split into features and target variable
X = dataset.drop('deposit', axis=1)
y = dataset['deposit']
original_columns = X.columns  # Save original column names for consistency

# Define numeric and categorical columns
numeric_cols = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous"
]

categorical_cols = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome"
]


# ColumnTransformer to scale numeric columns and one-hot encode categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'  # Pass through other columns if any
)

# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Convert splits back to DataFrame for ColumnTransformer compatibility
X_train = pd.DataFrame(X_train, columns=original_columns)
X_test = pd.DataFrame(X_test, columns=original_columns)

# Fit and transform training data, transform test data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Encode target labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Train Decision Tree classifier
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train_processed, y_train_encoded)

new_observation = pd.DataFrame({
    'age': [45],
    'job': ['admin.'],
    'marital': ['married'],
    'education': ['secondary'],
    'default': ['no'],
    'balance': [1000],
    'housing': ['yes'],
    'loan': ['no'],
    'contact': ['cellular'],
    'day': [5],
    'month': ['may'],
    'duration': [100],
    'campaign': [1],
    'pdays': [999],
    'previous': [0],
    'poutcome': ['unknown']
}, columns=original_columns)

# Preprocess the new observation
transformed_new_observation = preprocessor.transform(new_observation)

# Predict income class and decode
predicted_income_encoded = classifier.predict(transformed_new_observation)
predicted_income_label = le.inverse_transform(predicted_income_encoded)
print(f"\nPredicted income for the new observation: {predicted_income_label[0]}")

# Predict on test set
y_pred = classifier.predict(X_test_processed)

# Confusion matrix
cm = confusion_matrix(y_test_encoded, y_pred)
print("\nConfusion Matrix:")
print(cm)
print(f"True Negatives (TN): {cm[0, 0]}")
print(f"False Positives (FP): {cm[0, 1]}")
print(f"False Negatives (FN): {cm[1, 0]}")
print(f"True Positives (TP): {cm[1, 1]}")

# Accuracy score
accuracy = accuracy_score(y_test_encoded, y_pred)
print(f"\nAccuracy Score: {accuracy:.4f}")

# Compare predicted and actual values on test set
comparison_array = np.concatenate(
    (y_pred.reshape(-1, 1), y_test_encoded.reshape(-1, 1)), axis=1
)
print("\nPredicted vs Actual values (on test set):")
print(comparison_array)
