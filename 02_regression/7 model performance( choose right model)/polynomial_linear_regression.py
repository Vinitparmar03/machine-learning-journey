# polynomial_regression.py

# ====================================
# Polynomial Regression
# ====================================

# 1. Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 2. Importing the Dataset
# Replace with the actual dataset filename
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values  # Features
y = dataset.iloc[:, -1].values   # Target

# 3. Splitting the Dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# 4. Training the Polynomial Regression Model on the Training set
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Adjust the degree as needed
poly_reg = PolynomialFeatures(degree=4)
X_poly_train = poly_reg.fit_transform(X_train)

regressor = LinearRegression()
regressor.fit(X_poly_train, y_train)

# 5. Predicting the Test Set Results
X_poly_test = poly_reg.transform(X_test)
y_pred = regressor.predict(X_poly_test)

# Display predicted vs actual values
np.set_printoptions(precision=2)
print("Predicted vs Actual values (on test set):")
print(np.concatenate((y_pred.reshape(-1,1), y_test.reshape(-1,1)), axis=1))

# 6. Evaluating the Model Performance using R² Score
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")
