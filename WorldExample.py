import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class KMeans_Mat:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels = None

    def fit(self, X):
        # Randomly initialize centroids by selecting k random points from the dataset
        np.random.seed(42)
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        for i in range(self.max_iter):
            self.labels = self._assign_labels(X)  # Assign labels based on closest centroid
            new_centroids = self._compute_centroids(X)  # Recompute centroids

            # If centroids change very little, stop early (convergence condition)
            if self._centroid_shift(self.centroids, new_centroids) < self.tol:
                break
            self.centroids = new_centroids

    def _frobenius_norm(self, point, centroid):
        diff = point - centroid  # Difference between two points
        return (diff ** 2).sum() ** 0.5

    def _assign_labels(self, X):
        # Compute distances between each point and the centroids, using Frobenius norm
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i, centroid in enumerate(self.centroids):
            for j, point in enumerate(X):
                distances[j, i] = self._frobenius_norm(point, centroid)
        return np.argmin(distances, axis=1)

    def _compute_centroids(self, X):
        # Recompute the centroids based on the mean of the points assigned to each cluster
        new_centroids = []
        for i in range(self.n_clusters):
            cluster_points = X[self.labels == i]
            if len(cluster_points) > 0:
                new_centroids.append(np.mean(cluster_points, axis=0))
            else:
                new_centroids.append(self.centroids[i])  # If no points are assigned, retain the old centroid
        return np.array(new_centroids)

    def _centroid_shift(self, old_centroids, new_centroids):
        # Compute maximum change between old and new centroids
        shift = 0
        for old, new in zip(old_centroids, new_centroids):
            shift = max(shift, self._frobenius_norm(old, new))
        return shift

    def predict(self, X):
        # Assign new data points to the closest centroid
        return self._assign_labels(X)

    def elbow_method(self, X, max_k=10):
        inertia = []
        for k in range(1, max_k + 1):
            self.n_clusters = k
            self.fit(X)
            inertia.append(self._calculate_inertia(X))
        return inertia

    def _calculate_inertia(self, X):
        # Calculate inertia (sum of squared distances to closest centroid)
        inertia = 0
        for i in range(len(X)):
            inertia += self._frobenius_norm(X[i], self.centroids[self.labels[i]]) ** 2
        return inertia

# Load the dataset
data = pd.read_csv('Final.csv')

# Prepare the data
selected_data = data[['Entity', 'Year', 'Internet Users(%)', 'Broadband Subscription']]

# Filter out rows where both Internet Users(%) and Broadband Subscription are zero
selected_data = selected_data[(selected_data['Internet Users(%)'] > 0) | (selected_data['Broadband Subscription'] > 0)]

# Normalize the data to bring both features to the same scale
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(selected_data[['Internet Users(%)', 'Broadband Subscription']])

# Create KMeans instance
kmeans = KMeans_Mat()

# Perform elbow method to find the optimal number of clusters
inertia = kmeans.elbow_method(X_scaled, max_k=10)

# Determine the optimal number of clusters based on inertia
optimal_k = np.argmin(np.diff(np.diff(inertia))) + 2  # Simple method to find the elbow
print(f"Optimal number of clusters: {optimal_k}")

# Fit the model with the optimal number of clusters
kmeans.n_clusters = optimal_k
kmeans.fit(X_scaled)

# Add cluster labels to the DataFrame
selected_data['Cluster'] = kmeans.labels

print("\nClustering global Internet usage(%) per fixed broadband subscription (per 100 people) for countries each year.\n")
print("Here are the cluster assignments for each country and year based on their Internet usage and broadband subscription:")

print("\nNote: A higher cluster index (e.g., Cluster 2 or 3 in a 4-cluster model) generally means that a country in a given\nyear has higher internet usage (%) and/or better broadband infrastructure compared to countries in lower clusters.\nConversely, lower cluster indices represent countries with less developed internet infrastructure.\n")

# Adjust precision for printing
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Printing the results in a structured tabular format
print(selected_data[['Entity', 'Year', 'Internet Users(%)', 'Broadband Subscription', 'Cluster']].to_string(index=False))

selected_data.to_csv('clustered_internet_usage.csv', index=False)



