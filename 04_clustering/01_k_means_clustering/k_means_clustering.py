# =============================
# K-Means Clustering (with K-Means++)
# =============================

# 📌 Step 1: Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# 📌 Step 2: Load the dataset
dataset = pd.read_csv('Mall_Customers.csv')

# 📌 Step 3: Select features - here column 3 and 4 (Annual Income, Spending Score)
X = dataset.iloc[:, [3, 4]].values  # 2D feature space

# 📌 Step 4: Using the Elbow Method to find the optimal number of clusters (K)
WCSS = []  # Within-Cluster Sum of Squares

for i in range(1, 11):
    # init='k-means++' uses smart centroid initialization
    k_means = KMeans(n_clusters=i, init='k-means++', random_state=42)
    k_means.fit(X)
    WCSS.append(k_means.inertia_)  # inertia_ = WCSS for that K

# 📊 Plotting the Elbow Graph
plt.plot(range(1, 11), WCSS)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters (K)')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# 📌 Step 5: Train the K-Means model with the optimal K (assume K=5 from elbow)
k_means = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = k_means.fit_predict(X)  # returns cluster index for each point

# 📌 Step 6: Visualizing the clusters
# Each cluster plotted in a different color
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')

# Plotting the cluster centers
plt.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:, 1],
            s=300, c='yellow', label='Centroids')

plt.title('Clusters of Clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
