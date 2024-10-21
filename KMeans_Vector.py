import time

import numpy as np
import matplotlib.pyplot as plt

# K-Means clustering algorithm
class KMeans_Vector:
    def __init__(self, n_clusters=3, max_iter=300, tol=0.0001):
        self.centroids = None
        self.labels = None
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        # Randomly initialize centroids by selecting k random points from the dataset
        np.random.seed(42)
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        for i in range(self.max_iter):
            self.labels = self._assign_labels(X)  # Assign labels based on closest centroid
            new_centroids = self._compute_centroids(X)  # Recompute centroids

            # Check if the centroids have changed significantly, if not, stop
            if self._centroid_shift(self.centroids, new_centroids) < self.tol:
                break
            self.centroids = new_centroids

    def _assign_labels(self, X):
        # Compute distances from each point to each centroid and assign the closest centroid
        labels = []
        for x in X:
            distances = [self._euclidean_distance(x, centroid) for centroid in self.centroids]
            labels.append(distances.index(min(distances)))
        return np.array(labels)

    def _compute_centroids(self, X):
        # Compute the new centroids as the mean of the points assigned to each cluster
        new_centroids = []
        for i in range(self.n_clusters):
            cluster_points = X[self.labels == i]
            if len(cluster_points) > 0:
                new_centroids.append(np.mean(cluster_points, axis=0))
            else:
                new_centroids.append(self.centroids[i])  # Keep the old centroid if no points were assigned to the cluster
        return np.array(new_centroids)

    def _euclidean_distance(self, point1, point2):
        # Manually compute the Euclidean distance between two points
        distance = 0
        for p1, p2 in zip(point1, point2):
            distance += (p1 - p2) ** 2
        return distance ** 0.5

    def _centroid_shift(self, old_centroids, new_centroids):
        # Calculate the maximum shift between the old and new centroids
        shift = 0
        for old, new in zip(old_centroids, new_centroids):
            shift = max(shift, self._euclidean_distance(old, new))
        return shift

    def predict(self, X):
        # Assign new data points to the closest centroids
        return self._assign_labels(X)

# Sample data
np.random.seed(0)
X = np.random.rand(300, 2)

start_time = time.time()

# Applying KMeans
kmeans = KMeans_Vector(n_clusters=20)
kmeans.fit(X)
labels = kmeans.predict(X)

end_time = time.time()
print(f"K-Means clustering took {end_time - start_time:.4f} seconds")

# Plotting results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], s=300, c='red', marker='X')
plt.title("K-Means Clustering")
plt.show()

