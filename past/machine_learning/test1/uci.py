# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# import pandas as pd
#
#
# print("正在加载 yeast.csv 文件...")
#
# df = pd.read_csv('yeast.csv', header=None, delim_whitespace=True)
#
# if df.shape[1] > 9:
#     df = df.iloc[:, :9]
#
# X = df.iloc[:, 1:9].values
# X = X.astype(float)
#
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# k_range = range(2, 11)
# sse = []
#
# for k in k_range:
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, init='k-means++')
#     kmeans.fit(X_scaled)
#     sse.append(kmeans.inertia_)
#     print(f"K={k}, SSE={kmeans.inertia_:.4f}")
#
#
# plt.figure(figsize=(10, 6))
# plt.plot(k_range, sse, 'bo-', linewidth=2, markersize=8)
# plt.xlabel('Number of Clusters (K)', fontsize=12)
# plt.ylabel('Sum of Squared Errors (SSE)', fontsize=12)
# plt.title('Elbow Method for Optimal K on Yeast Dataset', fontsize=14)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.xticks(k_range)
# plt.show()

print("\n" + "="*50)
print("结果分析")
print("="*50)
print("1. 观察图表：")
print("   - 当 K 值较小时，增加聚类数量能显著降低 SSE（曲线陡峭）。")
print("   - 随着 K 值增大，SSE 的下降速度会变慢（曲线平缓）。")
print("   - '拐点' (Elbow Point) 就是曲线由陡峭变平缓的那个转折点。")
print("\n2. 确定最佳 K：")
print("   - 请观察图表，找到那个'胳膊肘'弯曲的位置。")
print("   - 例如，如果 K=4 之后曲线变得平缓，则最佳 K 值为 4。")
print("   - 这个 K 值代表了数据中自然存在的簇的数量，既能保证精度，又不会过度拟合。")

