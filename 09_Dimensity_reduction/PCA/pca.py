# -------------------------- Imports --------------------------
import numpy as np                        # Numerical computations
import matplotlib.pyplot as plt           # Plotting
import pandas as pd                       # Data handling
from sklearn.model_selection import train_test_split  # Splitting dataset
from sklearn.preprocessing import StandardScaler       # Feature scaling
from sklearn.decomposition import PCA                  # Principal Component Analysis
from sklearn.linear_model import LogisticRegression     # Classifier
from sklearn.metrics import confusion_matrix, accuracy_score  # Evaluation
from matplotlib.colors import ListedColormap           # For visualization

# -------------------------- Importing the dataset --------------------------
dataset = pd.read_csv('Wine.csv')
X = dataset.iloc[:, :-1].values    # Independent variables (features)
y = dataset.iloc[:, -1].values     # Dependent variable (target)

# -------------------------- Splitting the dataset --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# -------------------------- Feature Scaling --------------------------
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# -------------------------- Applying PCA --------------------------
pca = PCA(n_components=2)  # Reduce to 2 components for visualization
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# -------------------------- Training the Logistic Regression model --------------------------
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# -------------------------- Making the Confusion Matrix --------------------------
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# -------------------------- Visualising the Training set results --------------------------
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
    np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01)
)
plt.contourf(
    X1, X2,
    classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    alpha=0.75, cmap=ListedColormap(('red', 'green', 'blue'))
)
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        X_set[y_set == j, 0], X_set[y_set == j, 1],
        c=ListedColormap(('red', 'green', 'blue'))(i), label=j
    )
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# -------------------------- Visualising the Test set results --------------------------
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
    np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01)
)
plt.contourf(
    X1, X2,
    classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    alpha=0.75, cmap=ListedColormap(('red', 'green', 'blue'))
)
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        X_set[y_set == j, 0], X_set[y_set == j, 1],
        c=ListedColormap(('red', 'green', 'blue'))(i), label=j
    )
plt.title('Logistic Regression (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()
