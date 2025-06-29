# -------------------------- Imports --------------------------
import numpy as np                            # For numerical operations
import matplotlib.pyplot as plt               # For plotting
import pandas as pd                           # For data handling
from sklearn.model_selection import train_test_split  # For splitting dataset
from sklearn.preprocessing import StandardScaler       # For feature scaling
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA  # For LDA
from sklearn.linear_model import LogisticRegression     # For classifier
from sklearn.metrics import confusion_matrix, accuracy_score  # For evaluation
from matplotlib.colors import ListedColormap           # For colored plots

# -------------------------- Importing the dataset --------------------------
dataset = pd.read_csv('Wine.csv')
X = dataset.iloc[:, :-1].values    # Independent features
y = dataset.iloc[:, -1].values     # Target variable

# -------------------------- Splitting the dataset --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# -------------------------- Feature Scaling --------------------------
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# -------------------------- Applying LDA --------------------------
lda = LDA(n_components=2)  # Reduce to 2 Linear Discriminants
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

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
plt.xlabel('LD1')   # Linear Discriminant 1
plt.ylabel('LD2')   # Linear Discriminant 2
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
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.show()
