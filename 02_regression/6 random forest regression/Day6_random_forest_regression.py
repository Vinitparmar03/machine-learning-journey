# Random Forest Regression

# Importing the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# Assumes 'Position_Salaries.csv' contains columns like: 'Position', 'Level', 'Salary'
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values  # Selecting 'Level' as feature
y = dataset.iloc[:, -1].values    # Selecting 'Salary' as target

# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10000, random_state=0)  # Using 10,000 trees
regressor.fit(X, y)  # Fit model to the data

# Predicting a new result (e.g., salary for level 7)
predicted_salary = regressor.predict([[7]])
print(f"Predicted salary for level 7 is: {predicted_salary[0]}")

# Visualising the Random Forest Regression results (higher resolution for smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)  # Smaller step for finer resolution
X_grid = X_grid.reshape((len(X_grid), 1))  # Reshape for model input
plt.scatter(X, y, color='red')  # Actual data points
plt.plot(X_grid, regressor.predict(X_grid), color='blue')  # Predicted curve
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
