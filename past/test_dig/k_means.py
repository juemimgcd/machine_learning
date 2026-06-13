from sklearn.datasets import make_classification, make_blobs, make_regression
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(10,10))


# （1）随机产生1000个样本点，这些样本的需要利用至少3种不同的随机函数进行产生，再对随机生成的1000个点，
# 利用k均值聚类方法进行聚类，并对分类结果进行可视化作图；
# （2）利用自带的Kmeans函数针对iris数据和贷款数据进行聚类分析，并进行实验结果的分析，判断结果是否符合预期。
# （3）对“33门课程的考试成绩”进行聚类分析，分析的角度分别从课程和学生的两个维度出发
# ，并对分析结果进行解读，说明使用聚类可以达到何种的决策支持目标

x1, y1 = make_classification(n_samples=250,
                             n_features=2,
                             n_classes=2,
                             n_informative=2,
                             n_redundant=0,
                             random_state=20)
x2,y2 = make_blobs(
    n_samples=500,
    n_features=2,
    centers=2,
    random_state=20
)

x3,y3 = make_regression(n_samples=250,
                        n_features=2,
                        random_state=20)


x_completed = np.vstack([x1,x2,x3])

y_completed = np.concatenate([y1,y2,y3])


k_model = KMeans(n_clusters=3)

k_model.fit(x_completed)

y_pre = k_model.predict(x_completed)

labels = k_model.labels_

centers = k_model.cluster_centers_

ax1 = plt.subplot(221)
plt.scatter(x_completed[:, 0], x_completed[:, 1], c='gray')
plt.show()

ax2 = plt.subplot(222)

plt.scatter(x_completed[:, 0], x_completed[:, 1], c=y_pre)
plt.show()
# print(x_completed.shape,y_completed.shape)

