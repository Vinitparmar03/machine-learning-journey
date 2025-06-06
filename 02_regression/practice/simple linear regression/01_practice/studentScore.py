import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load and split
df = pd.read_csv('rounded_hours_student_scores.csv')
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=1)

# Train
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predict
y_predict = regressor.predict(X_test)

# Plot: Training set
plt.scatter(X_train, Y_train, color='red', label='Training data')
plt.plot(X_train, regressor.predict(X_train), color='blue', label='Regression line')
plt.title('Hours vs Percentage (Training Set)')
plt.xlabel('Hours')
plt.ylabel('Percentage')
plt.legend()
plt.show()

# Plot: Test set
plt.scatter(X_test, Y_test, color='green', label='Actual Test data')
plt.plot(X_test, y_predict, color='blue', label='Predicted Line (Test)')
plt.title('Hours vs Percentage (Test Set)')
plt.xlabel('Hours')
plt.ylabel('Percentage')
plt.legend()
plt.show()

# Compare actual vs predicted numerically (optional)
comparison = pd.DataFrame({'Actual': Y_test, 'Predicted': y_predict})
print(comparison.head(5))

# Metrics
mae = mean_absolute_error(Y_test, y_predict)
mse = mean_squared_error(Y_test, y_predict)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, y_predict)

# Print results
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R² Score:", r2)