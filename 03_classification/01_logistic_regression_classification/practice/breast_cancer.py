# Logistic Regression on Breast Cancer Dataset

# 1. Importing the libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# 2. Importing the dataset
df = pd.read_csv('breast_cancer.csv')
X = df.iloc[:, 1:-1].values  # Skipping ID column
y = df.iloc[:, -1].values    # Target: Diagnosis (2 = Benign, 4 = Malignant)

# 3. Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 4. Training the Logistic Regression model on the Training set
classifier = LogisticRegression(random_state=0, max_iter=2000)
classifier.fit(X_train, y_train)

# 5. Predicting the Test set results
y_pred = classifier.predict(X_test)
print("Predicted values on test set:")
print(y_pred)

# 6. Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# 7. Accuracy Score
acc = accuracy_score(y_test, y_pred)
print("\nAccuracy on Test Set:", acc)

# 8. k-Fold Cross Validation
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print("\n10-fold Cross Validation Accuracies:")
print(accuracies)
print("Mean Accuracy: {:.2f}".format(accuracies.mean()))
print("Standard Deviation: {:.2f}".format(accuracies.std()))
