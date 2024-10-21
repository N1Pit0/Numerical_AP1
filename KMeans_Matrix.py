import numpy as np
import time
from PIL import Image
from matplotlib import pyplot as plt


class KMeans_Matrix:
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


def load_image(image_path, size=(32, 32)):
    img = Image.open(image_path)
    img = img.resize(size)
    img_array = np.array(img)
    return img_array.reshape(-1, 3)  # Flatten image to 2D array of pixels


# Load the image and process it
image_path = 'Test_Images/car.jpg'  # Use your own image path here
X = load_image(image_path, size=(32, 32))  # Load and resize image to 32x32 pixels

# Apply K-Means clustering
start_time = time.time()

kmeans = KMeans_Matrix(n_clusters=20)
kmeans.fit(X)
labels = kmeans.predict(X)

end_time = time.time()
print(f"K-Means clustering took {end_time - start_time:.4f} seconds")

# Step 1: Create an empty image array with the same shape as the original image
clustered_image = np.zeros_like(X)

# Step 2: Assign colors based on the average color of each cluster
for i in range(kmeans.n_clusters):
    cluster_points = X[labels == i]  # Pixels in cluster i
    mean_color = np.mean(cluster_points, axis=0)  # Average color for the cluster
    clustered_image[labels == i] = mean_color  # Assign mean color to corresponding pixels

# Step 3: Reshape the clustered image back to original image dimensions
original_shape = (32, 32, 3)  # The original image was resized to 32x32 pixels
clustered_image = clustered_image.reshape(original_shape)

# Step 4: Plot the original and clustered images side by side
plt.figure(figsize=(10, 5))

# Plot the original image
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(X.reshape(original_shape))  # Reshape the flattened image back for display
plt.axis('off')

# Plot the clustered image
plt.subplot(1, 2, 2)
plt.title("Clustered Image")
plt.imshow(clustered_image.astype(int))  # Convert to integer values for display
plt.axis('off')

plt.show()
