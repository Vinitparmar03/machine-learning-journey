# MULTIPLE LINEAR REGRESSION - FULL CODE

# 1. Importing the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 2. Importing the dataset
# Replace the path if you're running on a different platform (e.g., local machine)
dataset = pd.read_csv('/content/sample_data/50_Startups.csv')

# Separating independent variables (X) and dependent variable (y)
X = dataset.iloc[:, :-1].values  # All rows, all columns except last (features)
y = dataset.iloc[:, -1].values   # All rows, last column (target)

# 3. Encoding the categorical data (State column)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# The State column is at index 3 (before encoding)
# After encoding, we convert X back to a NumPy array
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# 4. Avoiding the Dummy Variable Trap is handled automatically by most libraries now,
# but if needed, you can manually drop one dummy column (not done here)

# 5. Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 6. Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# 7. Predicting the Test set results
y_pred = regressor.predict(X_test)

# 8. Comparing predicted vs actual
print("Predicted vs Actual Profit on Test Set:\n")
for i in range(len(y_pred)):
    print(f"Predicted: {y_pred[i]:.2f}, Actual: {y_test[i]}")

# Optional: Visualize (if needed, for understanding)
# Scatter plot for predictions vs actual
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Line y=x
plt.title('Actual vs Predicted Profit')
plt.xlabel('Actual Profit')
plt.ylabel('Predicted Profit')
plt.grid(True)
plt.show()
