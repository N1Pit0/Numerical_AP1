import time

import numpy as np
import matplotlib.pyplot as plt

# K-Medoids clustering algorithm
class KMedoids_Vector:
    def __init__(self, n_clusters=3, max_iter=300, tol=0.0001):
        self.medoids = None
        self.labels = None
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        # Randomly initialize medoids by selecting k random points from the dataset
        np.random.seed(42)
        self.medoids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        for i in range(self.max_iter):
            self.labels = self._assign_labels(X)  # Assign labels based on closest medoid
            new_medoids = self._compute_medoids(X)  # Recompute medoids

            # Check if the medoids have changed significantly, if not, stop
            if self._medoid_shift(self.medoids, new_medoids) < self.tol:
                break
            self.medoids = new_medoids

    def _assign_labels(self, X):
        # Compute distances from each point to each medoid and assign the closest medoid
        labels = []
        for x in X:
            distances = [self._euclidean_distance(x, medoid) for medoid in self.medoids]
            labels.append(distances.index(min(distances)))
        return np.array(labels)

    def _compute_medoids(self, X):
        # Compute the new medoids by selecting the point in each cluster that minimizes the sum of distances
        new_medoids = []
        for i in range(self.n_clusters):
            cluster_points = X[self.labels == i]
            if len(cluster_points) > 0:
                new_medoids.append(self._find_medoid(cluster_points))
            else:
                new_medoids.append(self.medoids[i])  # Keep the old medoid if no points were assigned to the cluster
        return np.array(new_medoids)

    def _find_medoid(self, points):
        # Find the point in the cluster that minimizes the sum of distances to all other points in the cluster
        min_distance_sum = float('inf')
        medoid = points[0]
        for candidate in points:
            distance_sum = np.sum([self._euclidean_distance(candidate, p) for p in points])
            if distance_sum < min_distance_sum:
                min_distance_sum = distance_sum
                medoid = candidate
        return medoid

    def _euclidean_distance(self, point1, point2):
        # Manually compute the Euclidean distance between two points
        distance = 0
        for p1, p2 in zip(point1, point2):
            distance += (p1 - p2) ** 2
        return distance ** 0.5

    def _medoid_shift(self, old_medoids, new_medoids):
        # Calculate the maximum shift between the old and new medoids
        shift = 0
        for old, new in zip(old_medoids, new_medoids):
            shift = max(shift, self._euclidean_distance(old, new))
        return shift

    def predict(self, X):
        # Assign new data points to the closest medoids
        return self._assign_labels(X)

# Sample data
np.random.seed(0)
X = np.random.rand(300, 2)

start_time = time.time()

# Applying KMedoids
kmedoids = KMedoids_Vector(n_clusters=20)
kmedoids.fit(X)
labels = kmedoids.predict(X)

end_time = time.time()
print(f"K-Medoids clustering took {end_time - start_time:.4f} seconds")

# Plotting results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(kmedoids.medoids[:, 0], kmedoids.medoids[:, 1], s=300, c='red', marker='X')
plt.title("K-Medoids Clustering")
plt.show()
