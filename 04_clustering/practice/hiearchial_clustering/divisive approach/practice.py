# ------------------ Importing the Libraries ------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster

# ------------------ Importing the Dataset ------------------
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values  # Annual Income & Spending Score

# ------------------ Create the Linkage Matrix ------------------
Z = linkage(X, method='ward')  # Calculates hierarchical distances (top-down compatible)

# ------------------ Cutting the Tree Top-Down ------------------
# Set the desired number of clusters
k = 3

# fcluster cuts the dendrogram (top-down!) at a certain number of clusters
labels = fcluster(Z, t=k, criterion='maxclust')  # Cut to get `k` clusters

# ------------------ Visualising the Clusters ------------------
colors = ['red', 'blue', 'green', 'purple', 'orange']

for i in range(1, k + 1):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], 
                s=100, c=colors[i-1], label=f'Cluster {i}')

plt.title('Divisive Clustering (Top-Down using Linkage Matrix)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
