from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(10, 10))

iris = load_iris()

data = iris['data']
target = iris['target']

k_model = KMeans(n_clusters=3, random_state=30)

k_model.fit(data)

target_pre = k_model.predict(data)

plt.subplot(121)
plt.scatter(data[:, 0], data[:, 1], c=target)
plt.title('original')
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")

plt.subplot(122)
plt.scatter(data[:, 0], data[:, 1], c=target_pre)
plt.title('k_means')
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()

from sklearn.metrics import silhouette_score, adjusted_rand_score

silhouette = silhouette_score(data, target_pre)
ari = adjusted_rand_score(target, target_pre)

print(f"Silhouette Score: {silhouette}")
print(f"Adjusted Rand Index: {ari}")
