# Data Preprocessing Template

# 🚀 1. Importing the libraries
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# 📥 2. Importing the dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values  # Features
y = dataset.iloc[:, -1].values   # Target

# 🧹 3. Taking care of missing data (numeric columns only)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
x[:, 1:] = imputer.fit_transform(x[:, 1:])

# 🔤 4. Encoding categorical data
# 4.1 Encoding the Independent Variable (features)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

# 4.2 Encoding the Dependent Variable (target)
le = LabelEncoder()
y = le.fit_transform(y)

# ✂️ 5. Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# 📏 6. Feature Scaling
scaler = StandardScaler()
x_train[:, 3:] = scaler.fit_transform(x_train[:, 3:])  # Avoid scaling dummy variables
x_test[:, 3:] = scaler.transform(x_test[:, 3:])

# ✅ Final output to verify preprocessing
print("X_train:\n", x_train)
print("X_test:\n", x_test)
print("Y_train:\n", y_train)
print("Y_test:\n", y_test)
