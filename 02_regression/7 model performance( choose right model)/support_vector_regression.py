# svr_model.py

# ====================================
# Support Vector Regression (SVR)
# ====================================

# 1. Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 2. Importing the Dataset
dataset = pd.read_csv('Data.csv')  # Ensure Data.csv is in the same folder or update the path
X = dataset.iloc[:, :-1].values     # All columns except the last (features)
y = dataset.iloc[:, -1].values      # Last column (target)

# Reshape y to 2D array for StandardScaler
y = y.reshape(len(y), 1)

# 3. Splitting the Dataset into Training and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# 4. Feature Scaling (required for SVR)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()

X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)

# 5. Training the SVR model on the Training set
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')  # RBF = Radial Basis Function kernel (non-linear)
regressor.fit(X_train, y_train.ravel())  # y must be 1D array

# 6. Predicting the Test Set Results
y_pred_scaled = regressor.predict(sc_X.transform(X_test))  # Predict on scaled features
y_pred = sc_y.inverse_transform(y_pred_scaled.reshape(-1, 1))  # Convert predictions back to original scale

# Display predicted vs actual values
np.set_printoptions(precision=2)
print("Predicted vs Actual values (on test set):")
print(np.concatenate((y_pred, y_test), axis=1))

# 7. Evaluating the Model Performance using R² Score
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")
