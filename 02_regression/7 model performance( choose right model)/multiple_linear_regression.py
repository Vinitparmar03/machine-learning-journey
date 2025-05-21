# multiple_linear_regression.py

# ====================================
# Multiple Linear Regression
# ====================================

# 1. Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 2. Importing the Dataset
# Replace with your actual CSV file name
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values  # All columns except the last (independent variables)
y = dataset.iloc[:, -1].values   # Last column (dependent variable)

# 3. Splitting the Dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# 4. Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# 5. Predicting the Test Set Results
y_pred = regressor.predict(X_test)

# Display predicted vs actual values
np.set_printoptions(precision=2)
print("Predicted vs Actual values (on test set):")
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis=1))

# 6. Evaluating the Model Performance using R² Score
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")
