from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as KMeansPlus


iris = load_iris()
X_iris = iris.data
y_true = iris.target


X_vis = X_iris[:, :2]

print(f"{'算法':<15} | {'聚类数量':<10} | {'轮廓系数':<10}")
print("-" * 45)


km = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_km = km.fit_predict(X_iris)
score_km = silhouette_score(X_iris, labels_km)
print(f"{'K-Means':<15} | {len(set(labels_km)):<10} | {score_km:.4f}")


kmp = KMeansPlus(n_clusters=3, init='k-means++', random_state=42, n_init=10)
labels_kmp = kmp.fit_predict(X_iris)
score_kmp = silhouette_score(X_iris, labels_kmp)
print(f"{'K-Means++':<15} | {len(set(labels_kmp)):<10} | {score_kmp:.4f}")


dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_db = dbscan.fit_predict(X_iris)

n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)
score_db = silhouette_score(X_iris, labels_db) if n_clusters_db > 1 else -1
print(f"{'DBSCAN':<15} | {n_clusters_db:<10} | {score_db:.4f}")


plt.figure(figsize=(15, 4))

plt.subplot(1, 4, 1)
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_true, cmap='tab10')
plt.title('Ground Truth (Real Labels)')

plt.subplot(1, 4, 2)
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=labels_km, cmap='tab10')
plt.title('K-Means Result')

plt.subplot(1, 4, 3)
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=labels_kmp, cmap='tab10')
plt.title('K-Means++ Result')

plt.subplot(1, 4, 4)
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=labels_db, cmap='tab10')
plt.title('DBSCAN Result (eps=0.5, min=5)')

plt.tight_layout()
plt.show()