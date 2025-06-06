import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# Load dataset
dataset = pd.read_csv('Ice Cream.csv')  # Make sure this file is in your working directory or provide full path
X = dataset.iloc[:, :-1].values  # Feature(s) - Temperature
y = dataset.iloc[:, -1].values   # Target - Sales

# Split data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Linear Regression Model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)  # Train linear regression on training data

# Polynomial Features transformation (degree=4)
poly_reg = PolynomialFeatures(degree=4)
X_poly_train = poly_reg.fit_transform(X_train)  # Transform training data to polynomial features

# Polynomial Regression Model (linear regression on polynomial features)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly_train, y_train)  # Train polynomial regression on transformed data

# ---- Plotting ----

# Plot Linear Regression on training data
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, lin_reg.predict(X_train), color='blue')
plt.title('Linear Regression - Training Data')
plt.xlabel('Temperature')
plt.ylabel('Ice Cream Sales')
plt.show()

# Plot Linear Regression on test data
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, lin_reg.predict(X_test), color='blue')
plt.title('Linear Regression - Test Data')
plt.xlabel('Temperature')
plt.ylabel('Ice Cream Sales')
plt.show()

# Plot Polynomial Regression on training data
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, lin_reg_2.predict(X_poly_train), color='blue')
plt.title('Polynomial Regression (Degree 4) - Training Data')
plt.xlabel('Temperature')
plt.ylabel('Ice Cream Sales')
plt.show()

# Plot Polynomial Regression on test data
X_poly_test = poly_reg.transform(X_test)  # Transform test data to polynomial features (use transform, not fit_transform)
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, lin_reg_2.predict(X_poly_test), color='blue')
plt.title('Polynomial Regression (Degree 4) - Test Data')
plt.xlabel('Temperature')
plt.ylabel('Ice Cream Sales')
plt.show()


# Predict on training data using Linear Regression
y_train_pred_lin = lin_reg.predict(X_train)
# Predict on test data using Linear Regression
y_test_pred_lin = lin_reg.predict(X_test)

# Predict on training data using Polynomial Regression
y_train_pred_poly = lin_reg_2.predict(X_poly_train)
# Predict on test data using Polynomial Regression
y_test_pred_poly = lin_reg_2.predict(X_poly_test)

# Function to print all 4 metrics
def print_metrics(true, pred, dataset_name, model_name):
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    
    print(f"{model_name} performance on {dataset_name} data:")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  R^2 Score: {r2:.4f}\n")

# Linear Regression metrics
print_metrics(y_train, y_train_pred_lin, "Training", "Linear Regression")
print_metrics(y_test, y_test_pred_lin, "Test", "Linear Regression")

# Polynomial Regression metrics
print_metrics(y_train, y_train_pred_poly, "Training", "Polynomial Regression")
print_metrics(y_test, y_test_pred_poly, "Test", "Polynomial Regression")