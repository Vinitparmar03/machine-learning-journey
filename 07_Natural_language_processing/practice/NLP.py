# Importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk

# Downloading NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# Cleaning the texts
corpus = []
ps = PorterStemmer()
all_stopword = stopwords.words('english')
all_stopword.remove('not')

for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])  # Remove non-letters
    review = review.lower().split()  # Lowercase and split
    review = [ps.stem(word) for word in review if word not in all_stopword]  # Remove stopwords and stem
    corpus.append(' '.join(review))  # Join back into a string

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()  # Feature matrix
y = dataset.iloc[:, -1].values  # Labels

# Splitting dataset into Training and Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Training the Naive Bayes model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predicting Test set results
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Confusion Matrix and Accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print("Accuracy:", accuracy_score(y_test, y_pred))
