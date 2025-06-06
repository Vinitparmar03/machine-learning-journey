import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, 0:-1].values   # Features (all columns except last)
y = dataset.iloc[:, -1].values     # Target (last column)

# Reshape y to 2D array for scaler compatibility
y = y.reshape(len(y), 1)

# Feature scaling: scale both X and y to standard normal distribution
sc_X = StandardScaler()
sc_y = StandardScaler()
X_scaled = sc_X.fit_transform(X)
y_scaled = sc_y.fit_transform(y)

# Train SVR model with RBF kernel on scaled data
regressor = SVR(kernel='rbf')
regressor.fit(X_scaled, y_scaled.ravel())  # y_scaled.ravel() converts y to 1D array

# Optional: Predict for a single input value
value_to_predict = 0.6777231494691847
# 1. Scale input value
scaled_input = sc_X.transform([[value_to_predict]])
# 2. Predict on scaled input
scaled_prediction = regressor.predict(scaled_input).reshape(-1, 1)
# 3. Inverse scale prediction to original target scale
y_pred = sc_y.inverse_transform(scaled_prediction)

# Prepare data for plotting:

# 1. Convert scaled data back to original scale for visualization
X_original = sc_X.inverse_transform(X_scaled)  # shape: (n_samples, n_features)
y_original = sc_y.inverse_transform(y_scaled)  # shape: (n_samples, 1)

# 2. Sort data by feature values to ensure smooth plotting lines (only if 1 feature)
sorted_idx = X_original[:, 0].argsort()        # Indices that would sort the feature array
X_sorted = X_original[sorted_idx]               # Sorted features
X_scaled_sorted = X_scaled[sorted_idx]          # Corresponding scaled features sorted for prediction
# Predict on sorted scaled features and inverse transform predictions
y_pred_sorted = sc_y.inverse_transform(
    regressor.predict(X_scaled_sorted).reshape(-1, 1)
)

# Plot original data points and SVR prediction line (based on sorted data)
plt.scatter(X_original, y_original, color='red', label='Data')
plt.plot(X_sorted, y_pred_sorted, color='blue', label='Prediction')
plt.title('Features vs Target')
plt.xlabel('Features')
plt.ylabel('Target')
plt.legend()
plt.grid(True)
plt.show()

# Generate a fine grid of input values for smoother prediction curve (if 1 feature):
X_grid = np.arange(min(X_original), max(X_original), 0.01).reshape(-1, 1)

# Predict target values on this fine grid:
X_grid_scaled = sc_X.transform(X_grid)
y_grid_pred_scaled = regressor.predict(X_grid_scaled).reshape(-1, 1)
y_grid_pred = sc_y.inverse_transform(y_grid_pred_scaled)

# Plot original data and smooth prediction curve
plt.scatter(X_original, y_original, color='red', label='Data')
plt.plot(X_grid, y_grid_pred, color='blue', label='Smooth Prediction')
plt.title('Features vs Target (Smooth Curve)')
plt.xlabel('Features')
plt.ylabel('Target')
plt.legend()
plt.grid(True)
plt.show()

# Calculate performance metrics on entire dataset predictions:
y_pred_full_scaled = regressor.predict(X_scaled).reshape(-1, 1)
y_pred_full = sc_y.inverse_transform(y_pred_full_scaled)

r2 = r2_score(y_original, y_pred_full)
mse = mean_squared_error(y_original, y_pred_full)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_original, y_pred_full)

print(f"R² Score: {r2:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
