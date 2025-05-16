import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv('Real estate.csv')

# Features and target
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

# Polynomial transformation
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)

# Fit model
model = LinearRegression()
model.fit(X_poly, y)

# Predict values
y_pred = model.predict(X_poly)



# Compare actual vs predicted numerically (optional)
comparison = pd.DataFrame({'Actual': y, 'Predicted': y_pred})
print(comparison.head(10))  # Show first 10 comparisons

# Scatter plot: Actual vs Predicted
plt.scatter(y, y_pred, color='blue', edgecolor='k')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)  # Perfect prediction line
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.grid(True)
plt.show()







# Single new data point

new_data = np.array([[2012.917, 32, 84.87882, 10, 24.98298, 121.54024]])

# Transform the new data using the same polynomial transformer
new_data_poly = poly.transform(new_data)

# Predict using the trained model
new_prediction = model.predict(new_data_poly)

print(f"Predicted house price for the new data point: {new_prediction}")

