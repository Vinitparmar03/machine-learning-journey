# svr_salary_prediction.py
# Support Vector Regression (SVR) Model for Predicting Salaries based on Position Level

# -------------------------------
# 1. Importing Required Libraries
# -------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# ---------------------
# 2. Loading the Dataset
# ---------------------
# Ensure 'Position_Salaries.csv' is in the same directory as this script
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values  # Independent variable (Position level)
y = dataset.iloc[:, -1].values    # Dependent variable (Salary)

# ---------------------
# 3. Reshaping the Target
# ---------------------
# SVR expects 2D input for y, so reshape it
y = y.reshape(len(y), 1)

# -------------------------
# 4. Feature Scaling (SVR requires scaling)
# -------------------------
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_Y.fit_transform(y)

# ----------------------------------
# 5. Training the SVR Model on the Data
# ----------------------------------
regressor = SVR(kernel='rbf')  # Using Radial Basis Function kernel
regressor.fit(X, y.ravel())    # Flatten y for SVR training

# --------------------------------------
# 6. Predicting a New Result (Level = 6.5)
# --------------------------------------
predicted_scaled = regressor.predict(sc_X.transform([[6.5]]))
predicted_salary = sc_Y.inverse_transform(predicted_scaled.reshape(-1, 1))
print("Predicted salary for level 6.5:", predicted_salary[0][0])

# -------------------------------------
# 7. Visualising the SVR Results (Basic)
# -------------------------------------
plt.scatter(sc_X.inverse_transform(X), sc_Y.inverse_transform(y), color='red', label='Actual')
plt.plot(sc_X.inverse_transform(X), sc_Y.inverse_transform(regressor.predict(X).reshape(-1, 1)),
         color='blue', label='SVR Prediction')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------------------------------
# 8. Visualising the SVR Results (High Resolution Curve)
# ----------------------------------------------------
X_grid = np.arange(min(sc_X.inverse_transform(X)),
                   max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_Y.inverse_transform(y), color='red', label='Actual')
plt.plot(X_grid,
         sc_Y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1, 1)),
         color='blue', label='SVR Curve')
plt.title('Truth or Bluff (SVR - High Resolution)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend()
plt.grid(True)
plt.show()
