import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load the dataset from CSV file
dataset = pd.read_csv('forestfires.csv')

# Convert categorical columns ('month' and 'day') to numerical using one-hot encoding
dataset_encoded = pd.get_dummies(dataset, columns=['month', 'day'], drop_first=True)

# Split features (X) and target variable (y = area affected by fire)
X = dataset_encoded.iloc[:, :-1].values   # All columns except last one
y = dataset_encoded.iloc[:, -1].values    # Last column is the target: 'area'

# Standardize the feature matrix and target vector
sc_X = StandardScaler()
sc_y = StandardScaler()
X_scaled = sc_X.fit_transform(X)
y_scaled = sc_y.fit_transform(y.reshape(-1, 1))

# Initialize and train SVR model with RBF kernel
regressor = SVR(kernel='rbf')
regressor.fit(X_scaled, y_scaled.ravel())

# Make a prediction using the first sample (for example)
sample_input = X[0].reshape(1, -1)                        # Reshape to 2D as model expects
sample_input_scaled = sc_X.transform(sample_input)       # Scale the input
sample_pred_scaled = regressor.predict(sample_input_scaled).reshape(-1, 1)  # Predict
sample_pred = sc_y.inverse_transform(sample_pred_scaled) # Inverse transform to get actual scale
print(f"Single prediction for first sample (actual={y[0]}): {sample_pred[0, 0]}")

# Predict target values for the entire dataset
y_pred_scaled = regressor.predict(X_scaled).reshape(-1, 1)
y_pred = sc_y.inverse_transform(y_pred_scaled)

# Set decimal precision for printing, then display predicted and actual values side-by-side for comparison
np.set_printoptions(precision=2)
print("Predicted vs Actual values (on test set):")
print(np.concatenate((y_pred, y.reshape(-1, 1)), axis=1))

# Flatten actual and predicted target values for plotting
y_flat = y.ravel() 
y_pred_flat = y_pred.ravel()

# Plot 1: Basic scatter plot of Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_flat, y_pred_flat, color='blue', alpha=0.6, label='Predicted vs Actual')

# Add a red dashed line representing perfect prediction (y = x)
lims = [min(min(y_flat), min(y_pred_flat)), max(max(y_flat), max(y_pred_flat))]
plt.plot(lims, lims, 'r--', label='Perfect prediction (y = x)')

# Add axis labels and title
plt.xlabel('Actual Area')
plt.ylabel('Predicted Area')
plt.title('Actual vs Predicted Forest Fire Area')
plt.legend()
plt.grid(True)
plt.show()

# Evaluate model performance using standard regression metrics
r2 = r2_score(y, y_pred)                            # Coefficient of determination
mse = mean_squared_error(y, y_pred)                 # Mean Squared Error
rmse = np.sqrt(mse)                                 # Root Mean Squared Error
mae = mean_absolute_error(y, y_pred)                # Mean Absolute Error

# Print all evaluation metrics
print(f"R² Score: {r2:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
