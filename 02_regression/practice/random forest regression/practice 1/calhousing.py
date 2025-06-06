# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset from a CSV file
dataset = pd.read_csv('cal_housing.csv')

# Separate features (X) and target variable (y)
X = dataset.iloc[:, :-1].values  # All columns except the last (features)
y = dataset.iloc[:, -1].values   # Last column (target variable - median house value)

# Split the dataset into training and testing sets (90% train, 10% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Initialize the Random Forest Regressor with 20 decision trees
regressor = RandomForestRegressor(n_estimators=10, random_state=0)

# Train the model using the training data
regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regressor.predict(X_test)

# Display predicted and actual values side-by-side
print("Predicted vs Actual values (on test set):")
# Reshape both arrays to (n,1) to stack them horizontally
print(np.concatenate((y_pred.reshape(-1, 1), y_test.reshape(-1, 1)), axis=1))

# Evaluate the model using common regression metrics
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))              # Average absolute difference
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))                # Average squared difference
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))  # Square root of MSE
print("R-squared:", r2_score(y_test, y_pred))                                   # Goodness of fit (1 is perfect)

# Plot predicted vs actual values
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Actual vs Predicted values')
plt.grid(True)
plt.show()

# Make a single prediction using a custom input
# Feature order must match dataset: [longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income]
# print(regressor.predict([[-122.23, 37.88, 41, 880, 129, 322, 126, 8.3252]]))
