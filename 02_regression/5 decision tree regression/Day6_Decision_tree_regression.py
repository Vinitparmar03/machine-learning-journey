# Decision Tree Regression

# Importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
# Assumes 'Position_Salaries.csv' has columns like: 'Position', 'Level', 'Salary'
df = pd.read_csv('Position_Salaries.csv')
X = df.iloc[:, 1:-1].values  # Selecting 'Level' as feature (2nd column)
y = df.iloc[:, -1].values    # Selecting 'Salary' as label (last column)

# Training the Decision Tree Regression model on the whole dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)  # Fitting model to data

# Predicting a new result (e.g., for level 6.5)
predicted_salary = regressor.predict([[6.5]])
print(f"Predicted salary for level 6.5 is: {predicted_salary[0]}")

# Visualising the Decision Tree Regression results (higher resolution)
x_grid = np.arange(min(X), max(X), 0.1)  # Creating a grid for high-res plot
x_grid = x_grid.reshape((len(x_grid), 1))  # Reshape to correct input format
plt.scatter(X, y, color='red')  # Actual data points
plt.plot(x_grid, regressor.predict(x_grid), color='blue')  # Predicted curve
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
