import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix

tf.__version__

df = pd.read_csv("Churn_Modelling.csv")
X = df.iloc[:, 3:-1].values
y = df.iloc[:, -1].values

le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [1])], remainder="passthrough")
X = np.array(ct.fit_transform(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu', ))
ann.add(tf.keras.layers.Dense(units=6, activation='relu', ))

ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

ann.fit(X_train, y_train, batch_size=32, epochs=100)

y_pred = (ann.predict(sc.transform([[1,0,0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5).astype(int)

print(y_pred)

y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis=1))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print("Accuracy:", accuracy_score(y_test, y_pred))


