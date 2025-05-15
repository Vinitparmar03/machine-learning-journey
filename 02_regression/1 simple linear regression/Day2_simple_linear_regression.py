# Simple Linear Regression

# 📌 Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 📌 Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values  # Independent variable (Years of Experience)
y = dataset.iloc[:, -1].values   # Dependent variable (Salary)

# 📌 Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# 📌 Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# 📌 Predicting the Test set results
y_predict = regressor.predict(x_test)

# 📌 Visualising the Training set results
plt.scatter(x_train, y_train, color='red')  # Actual data points
plt.plot(x_train, regressor.predict(x_train), color='blue')  # Regression line
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# 📌 Visualising the Test set results
plt.scatter(x_test, y_test, color='red')  # Actual test data
plt.plot(x_train, regressor.predict(x_train), color='blue')  # Same regression line
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
