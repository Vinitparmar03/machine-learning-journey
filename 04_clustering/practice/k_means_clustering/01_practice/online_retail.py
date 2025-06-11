import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# ------------------ Load Dataset ------------------
df = pd.read_csv('OnlineRetail.csv', encoding='ISO-8859-1')

# ------------------ Clean and Filter ------------------
df.dropna(subset=['Quantity', 'UnitPrice', 'CustomerID'], inplace=True)
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df = df[(df['Quantity'] < 10000) ]

# ------------------ Feature Engineering ------------------
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# ------------------ Group by Customer ------------------
df_grouped = df.groupby('CustomerID').agg({
    'Quantity': 'sum',
    'UnitPrice': 'mean',
    'TotalPrice': 'sum'
}).reset_index()

# ------------------ Feature Selection ------------------
X = df_grouped[['Quantity', 'UnitPrice', 'TotalPrice']].values

# Optional: Log transform
X = np.log1p(X)

# ------------------ Elbow Method ------------------
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# ------------------ Apply KMeans ------------------
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# ------------------ Visualize Clusters ------------------
# Only 2D projection using first 2 features
plt.figure(figsize=(10, 6))
colors = ['red', 'green', 'blue']
for cluster_id in range(optimal_k):
    plt.scatter(
        X[y_kmeans == cluster_id, 0],
        X[y_kmeans == cluster_id, 2],  # Quantity vs TotalPrice
        s=50,
        c=colors[cluster_id],
        label=f'Cluster {cluster_id + 1}'
    )

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 2],
    s=200,
    c='yellow',
    marker='X',
    label='Centroids'
)

plt.title('Improved KMeans Clustering (Quantity vs TotalPrice)')
plt.xlabel('Log Quantity')
plt.ylabel('Log Total Price')
plt.legend()
plt.grid(True)
plt.show()
