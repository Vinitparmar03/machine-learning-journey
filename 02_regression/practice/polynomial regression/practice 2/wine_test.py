import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load dataset (semicolon-separated CSV)
df = pd.read_csv('winequality-white.csv', sep=';')

# Step 2: Separate independent variables (X) and target variable (y)
X = df.iloc[:, :-1].values  # All features except 'quality'
y = df.iloc[:, -1].values   # 'quality' column (target)

# Step 3: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# Step 4: Transform features into polynomial features (degree 4)
poly_reg = PolynomialFeatures(degree=4)
X_poly_train = poly_reg.fit_transform(X_train)
X_poly_test = poly_reg.transform(X_test)  # Also transform test set

# Step 5: Fit linear regression model on polynomial features
lin_reg = LinearRegression()
lin_reg.fit(X_poly_train, y_train)

# Step 6: Make predictions
y_train_pred = lin_reg.predict(X_poly_train)
y_test_pred = lin_reg.predict(X_poly_test)

# Step 7: Evaluate performance
print("Training R² Score:", r2_score(y_train, y_train_pred))
print("Test R² Score:", r2_score(y_test, y_test_pred))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))

# Step 8: Visualization - Predicted vs Actual for Test Data
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title("Polynomial Regression: Actual vs Predicted (Test Set)")
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")
plt.grid(True)
plt.show()


comparison_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_test_pred
})

# Print first 10 rows to check predictions vs actual
print(comparison_df.head(100))