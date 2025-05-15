# POLYNOMIAL REGRESSION - FULL CODE

# 1. Importing the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 2. Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')  # Make sure the CSV file is in the working directory
X = dataset.iloc[:, 1:-1].values   # Position level (independent variable)
y = dataset.iloc[:, -1].values     # Salary (dependent variable)

# 3. Training the Linear Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# 4. Training the Polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=6)  # You can change the degree as needed
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# 5. Visualising the Linear Regression results
plt.scatter(X, y, color='red')  # Actual data points
plt.plot(X, lin_reg.predict(X), color='blue')  # Prediction line
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# 6. Visualising the Polynomial Regression results
plt.scatter(X, y, color='red')  # Actual data
plt.plot(X, lin_reg_2.predict(X_poly), color='blue')  # Polynomial regression curve
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# 7. Visualising the Polynomial Regression results (higher resolution curve)
X_grid = np.arange(min(X), max(X), 0.1)  # Smoother curve with smaller steps
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression - High Resolution)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# 8. Predicting a new result with Linear Regression
linear_pred = lin_reg.predict([[6.5]])
print(f"Linear Regression prediction for 6.5: {linear_pred[0]}")

# 9. Predicting a new result with Polynomial Regression
poly_pred = lin_reg_2.predict(poly_reg.fit_transform([[10]]))
print(f"Polynomial Regression prediction for 10: {poly_pred[0]}")

# Optional: Print all y values for reference
print("Actual salary values in dataset:", y)
