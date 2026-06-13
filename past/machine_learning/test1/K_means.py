import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


class MyKMeans:
    def __init__(self, k=3, max_iters=300, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels_ = None
        self.sse_history = []

    def _init_centroids(self, X):
        np.random.seed(42)  # 固定随机种子以便复现
        indices = np.random.choice(X.shape[0], self.k, replace=False)
        return X[indices]

    def _euclidean_distance(self, x1, x2):
        if x1.ndim == 1 and x2.ndim == 1:
            return np.sqrt(np.sum((x1 - x2) ** 2))
        else:
            return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))

    def _assign_clusters(self, X, centroids):

        labels = []
        for x in X:
            distances = [self._euclidean_distance(x, c) for c in centroids]
            labels.append(np.argmin(distances))
        return np.array(labels)

    def _update_centroids(self, X, labels):

        new_centroids = []
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids.append(np.mean(cluster_points, axis=0))
            else:

                new_centroids.append(X[np.random.choice(X.shape[0])])
        return np.array(new_centroids)

    def _calculate_sse(self, X, labels, centroids):
        sse = 0
        for i in range(len(X)):
            sse += self._euclidean_distance(X[i], centroids[labels[i]]) ** 2
        return sse

    def fit(self, X):
        self.centroids = self._init_centroids(X)

        for i in range(self.max_iters):
            self.labels_ = self._assign_clusters(X, self.centroids)

            current_sse = self._calculate_sse(X, self.labels_, self.centroids)
            self.sse_history.append(current_sse)

            new_centroids = self._update_centroids(X, self.labels_)

            centroid_shift = np.sum(
                [self._euclidean_distance(self.centroids[j], new_centroids[j]) for j in range(self.k)])
            self.centroids = new_centroids

            if centroid_shift < self.tol:
                print(f"迭代 {i + 1} 次后收敛。")
                break

        return self


X_samples, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)


k = 4
my_kmeans = MyKMeans(k=k)
my_kmeans.fit(X_samples)


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_samples[:, 0], X_samples[:, 1], c=my_kmeans.labels_, s=30, cmap='viridis')
plt.scatter(my_kmeans.centroids[:, 0], my_kmeans.centroids[:, 1], c='red', s=200, alpha=0.75, marker='X',
            label='Centroids')
plt.title(f'My K-Means Clustering (K={k})')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(range(1, len(my_kmeans.sse_history) + 1), my_kmeans.sse_history, marker='o')
plt.xlabel('Iterations')
plt.ylabel('SSE')
plt.title('SSE Convergence')

plt.tight_layout()
plt.show()
