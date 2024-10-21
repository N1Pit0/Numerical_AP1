import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time


class KMedoids_Matrix:
    def __init__(self, n_clusters=3, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.medoids = None
        self.labels = None

    def fit(self, X):
        np.random.seed(42)
        # Randomly initialize medoids
        self.medoids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        for _ in range(self.max_iter):
            self.labels = self._assign_labels(X)
            new_medoids = self._compute_medoids(X)

            if np.array_equal(new_medoids, self.medoids):
                break
            self.medoids = new_medoids

    def _frobenius_norm(self, point, centroid):
        diff = point - centroid  # Difference between two matrices
        return (diff ** 2).sum() ** 0.5

    def _assign_labels(self, X):
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i, medoid in enumerate(self.medoids):
            # Calculate Frobenius norm between each point and medoid
            distances[:, i] = np.array([self._frobenius_norm(point, medoid) for point in X])
        return np.argmin(distances, axis=1)

    def _compute_medoids(self, X):
        new_medoids = []
        for i in range(self.n_clusters):
            cluster_points = X[self.labels == i]
            if len(cluster_points) > 0:
                # Calculate sum of distances within the cluster and find the new medoid
                distances = np.sum([[self._frobenius_norm(p1, p2) for p2 in cluster_points] for p1 in cluster_points], axis=1)
                new_medoids.append(cluster_points[np.argmin(distances)])
            else:
                new_medoids.append(self.medoids[i])
        return np.array(new_medoids)

    def predict(self, X):
        return self._assign_labels(X)


def load_image(image_path, size=(32, 32)):
    img = Image.open(image_path)
    img = img.resize(size)
    img_array = np.array(img).astype(float)
    return img_array  # Return original shape, do not flatten


# Load an image and process it
image_path = 'Test_Images/car.jpg'  # Path to your image file

X_original = load_image(image_path, size=(32, 32))
X_flattened = X_original.reshape(-1, 3)  # Flatten to a 2D array where each row is a pixel

# Applying K-Medoids
start_time = time.time()

kmedoids = KMedoids_Matrix(n_clusters=20)
kmedoids.fit(X_flattened)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"K-Medoids clustering took {elapsed_time:.4f} seconds")

labels = kmedoids.predict(X_flattened)

# Visualize the results
clustered_image = np.zeros_like(X_flattened)

for i in range(kmedoids.n_clusters):
    mean_color = np.mean(X_flattened[labels == i], axis=0)  # Average color for the cluster
    clustered_image[labels == i] = mean_color  # Assign mean color to the corresponding pixels

# Reshape the clustered image back to the original image dimensions
clustered_image = clustered_image.reshape((32, 32, 3))

# Plot original and clustered images
plt.figure(figsize=(10, 5))

# Display the original image
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(X_original.astype(int))  # No need to reshape, already in the correct shape
plt.axis('off')

# Display the clustered image
plt.subplot(1, 2, 2)
plt.title("Clustered Image")
plt.imshow(clustered_image.astype(int))  # Convert to int for displaying
plt.axis('off')

plt.show()
