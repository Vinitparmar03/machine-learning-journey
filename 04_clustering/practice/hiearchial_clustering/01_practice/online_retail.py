import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# ------------------ Load and Clean Dataset ------------------
df = pd.read_csv('OnlineRetail.csv', encoding='ISO-8859-1')
df.dropna(subset=['Quantity', 'UnitPrice', 'CustomerID'], inplace=True)
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df = df[df['Quantity'] < 10000]  # Optional: Remove outliers

# ------------------ Feature Engineering ------------------
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# ------------------ Aggregate by Customer ------------------
df_grouped = df.groupby('CustomerID').agg({
    'Quantity': 'sum',
    'UnitPrice': 'mean',
    'TotalPrice': 'sum'
}).reset_index()

# ------------------ Select Features ------------------
X = df_grouped[['Quantity', 'UnitPrice', 'TotalPrice']].values

# ------------------ Optional: Log Transform ------------------
X = np.log1p(X)  # To reduce skew

# ------------------ Dendrogram to Find Optimal Clusters ------------------
plt.figure(figsize=(10, 6))
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram (Hierarchical Clustering)')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.grid(True)
plt.show()

# ------------------ Apply Hierarchical Clustering ------------------
optimal_clusters = 3  # Set based on dendrogram
hc = AgglomerativeClustering(n_clusters=optimal_clusters, metric='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

# ------------------ Visualize Clusters ------------------
plt.figure(figsize=(10, 6))
colors = ['red', 'green', 'blue', 'cyan', 'magenta']
for cluster_id in range(optimal_clusters):
    plt.scatter(
        X[y_hc == cluster_id, 0],  # Quantity (log)
        X[y_hc == cluster_id, 2],  # TotalPrice (log)
        s=50,
        c=colors[cluster_id],
        label=f'Cluster {cluster_id + 1}'
    )

plt.title('Hierarchical Clustering (Quantity vs TotalPrice)')
plt.xlabel('Log Quantity')
plt.ylabel('Log Total Price')
plt.legend()
plt.grid(True)
plt.show()
