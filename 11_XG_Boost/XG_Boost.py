# 📦 Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from xgboost import XGBClassifier

# 📥 Importing the dataset
dataset = pd.read_csv('Data (3).csv')
X = dataset.iloc[:, :-1].values      # Features
y = dataset.iloc[:, -1].values       # Target variable

# 🔁 Relabel target classes from [2, 4] to [0, 1] (if binary classification)
y = np.where(y == 2, 0, 1)

# ✂️ Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 🧠 Training the XGBoost model on the Training set
classifier = XGBClassifier(eval_metric='logloss')  # Removed use_label_encoder to fix warning
classifier.fit(X_train, y_train)

# ✅ Making predictions and evaluating using a Confusion Matrix
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
print("Accuracy on Test Set: {:.2f} %".format(accuracy * 100))

# 🔁 Applying k-Fold Cross Validation
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print("Mean Accuracy (CV): {:.2f} %".format(accuracies.mean() * 100))
print("Standard Deviation (CV): {:.2f} %".format(accuracies.std() * 100))
