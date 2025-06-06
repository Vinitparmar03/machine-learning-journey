import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load dataset
df = pd.read_csv('insurance.csv')

# Separate features and target
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

# OneHotEncode categorical columns (indices 1, 4, 5)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1, 4, 5])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Split into training and test sets (80%-20%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Initialize StandardScaler
scaler = StandardScaler()

# Fit scaler on training data and transform training data
X_train = scaler.fit_transform(X_train)

# Transform test data with the same scaler (do NOT fit again!)
X_test = scaler.transform(X_test)

# Create and train Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predict on test set
y_pred = regressor.predict(X_test)

# Compare actual and predicted values
comparison = pd.DataFrame({'Actual': Y_test, 'Predicted': y_pred})
print(comparison.head(10))

# Plot actual vs predicted values
plt.scatter(Y_test, y_pred, color='blue')
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red')  # Perfect prediction line y=x
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



input_features = [60, 'female', 25.84, 0, 'no', 'northwest']  

# Convert input_features into numpy array and reshape for one sample (1 row)
input_array = np.array(input_features).reshape(1, -1)  

# Apply the same ColumnTransformer 'ct' to encode categorical features
input_encoded = ct.transform(input_array)  

# Apply the same StandardScaler 'scaler' to scale features
input_scaled = scaler.transform(input_encoded)  

# Predict the target value using the trained model
predicted_value = regressor.predict(input_scaled)

# Print the predicted value in one line with comment
print(f"Predicted value for input features {input_features} is: {predicted_value[0]:.2f}")
