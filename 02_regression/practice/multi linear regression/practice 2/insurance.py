import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load dataset from CSV file
df = pd.read_csv('insurance.csv')

# Check if there are any missing values in the dataframe columns (returns True/False for each column)
# print(df.isnull().any())

# Separate independent variables (features) and dependent variable (target)
X = df.iloc[:, :-1].values  # All columns except the last one as features
Y = df.iloc[:, -1].values   # Last column as target variable

# Apply OneHotEncoding to categorical columns at indices 1, 4, and 5
# 'remainder=passthrough' means other columns will remain unchanged
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1, 4, 5])], remainder='passthrough')
X = np.array(ct.fit_transform(X))  # Transform features and convert result to numpy array

from sklearn.model_selection import train_test_split

# Split dataset into training set (80%) and test set (20%)
# random_state=0 ensures reproducibility
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression

# Create a LinearRegression model instance
regressor = LinearRegression()

# Train the model on training data
regressor.fit(X_train, Y_train)

# Predict target values for the test set
y_pred = regressor.predict(X_test)

# Optional: print first 10 predicted vs actual values side by side
comparison = pd.DataFrame({'Actual': Y_test, 'Predicted': y_pred})
print(comparison.head(10))

# Plot actual vs predicted values
plt.scatter(Y_test, y_pred, color='blue')  # Scatter plot: predicted vs actual
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red')  # Reference line y = x (perfect predictions)
plt.title('Actual vs Predicted Profit')
plt.xlabel('Actual Profit')
plt.ylabel('Predicted Profit')
plt.grid(True)
plt.show()

# y_test: actual values
# y_pred: predicted values by the model

# 1. Mean Absolute Error (MAE)
# Average of absolute differences between actual and predicted values
mae = mean_absolute_error(Y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# 2. Mean Squared Error (MSE)
# Average of squared differences between actual and predicted values
mse = mean_squared_error(Y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.4f}")

# 3. Root Mean Squared Error (RMSE)
# Square root of MSE, gives error in original units
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# 4. R-squared (R²) Score
# Proportion of variance in dependent variable explained by the model (1 is perfect)
r2 = r2_score(Y_test, y_pred)
print(f"R-squared (R²) Score: {r2:.4f}")
