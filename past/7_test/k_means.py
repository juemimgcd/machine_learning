from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(12, 10))

x, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
x_scaled = StandardScaler().fit_transform(x)

k_model = KMeans(n_clusters=2, random_state=42)
k_model.fit(x_scaled)
k_labels = k_model.predict(x_scaled)
k_centers = k_model.cluster_centers_

d_model = DBSCAN(eps=0.2, min_samples=5)
d_model.fit(x_scaled)
d_labels = d_model.labels_

dbscan_centers = []
if len(np.unique(d_labels)) > 1:
    for label in np.unique(d_labels):
        if label != -1:
            cluster_points = x_scaled[d_labels == label]
            dbscan_centers.append(cluster_points.mean(axis=0))

plt.subplot(121)
plt.scatter(x_scaled[:, 0], x_scaled[:, 1], c=k_labels)
plt.scatter(k_centers[:, 0], k_centers[:, 1], c='red')
plt.title('kmeans')

plt.subplot(122)
plt.scatter(x_scaled[:, 0], x_scaled[:, 1], c=d_labels, cmap='viridis', s=10)
if len(dbscan_centers) > 0:
    dbscan_centers = np.array(dbscan_centers)
    plt.scatter(dbscan_centers[:, 0], dbscan_centers[:, 1], c='red', marker='x', s=100)
plt.title('dbscan')
plt.show()
