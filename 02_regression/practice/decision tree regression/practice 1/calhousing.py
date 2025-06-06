import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset from CSV file
dataset = pd.read_csv('cal_housing.csv')

# Separate features (X) and target variable (y)
X = dataset.iloc[:, :-1].values  # All columns except last as features
y = dataset.iloc[:, -1].values   # Last column as target

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize Decision Tree Regressor with a fixed random state for reproducibility
regressor = DecisionTreeRegressor(nestimator=10,random_state=0)

# Train the regressor on the training data
regressor.fit(X_train, y_train)

# Predict target values for the test data
y_pred = regressor.predict(X_test)

# Display predicted and actual values side-by-side
print("Predicted vs Actual values (on test set):")
# Reshape 1D arrays to 2D column vectors for concatenation along columns (axis=1)
print(np.concatenate((y_pred.reshape(-1, 1), y_test.reshape(-1, 1)), axis=1))

# Evaluate the model performance using common regression metrics
print('Mean Absolute Error (MAE):', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error (MSE):', mean_squared_error(y_test, y_pred))
print('R-squared (R2) Score:', r2_score(y_test, y_pred))

# Optional: Visualize predicted vs actual values using scatter plot
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Decision Tree Regression: Predicted vs Actual')
plt.show()
