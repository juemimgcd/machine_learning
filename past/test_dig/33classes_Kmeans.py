import numpy as np
import pandas as pd
from sklearn.cluster import KMeans,DBSCAN
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(10,10))

# （3）对“33门课程的考试成绩”进行聚类分析，分析的角度分别从课程和学生的两个维度出发
# ，并对分析结果进行解读，说明使用聚类可以达到何种的决策支持目标

df = pd.read_csv('33classes_scores.csv', encoding='gbk')

data = df[['学号', '课程代码', '期末成绩']]
data = data.fillna(value=0)

course_dig = data.groupby('课程代码')['期末成绩'].agg(['mean','min','max']).reset_index()
print(course_dig.head())

scaler = StandardScaler()
course_dig_scaled = scaler.fit_transform(course_dig[['mean','min','max']])

scores1 = []

# for i in range(1,10):
#     k_model = KMeans(n_clusters=i,random_state=30)
#     k_model.fit(course_dig_scaled)
#     scores1.append(k_model.inertia_)
#
# plt.plot(range(1, 10), scores1, marker='o')
# plt.xlabel('Number of clusters')
# plt.ylabel('Inertia')
# plt.title('Elbow Method for Optimal k (Courses)')
# plt.show()

# k_model = KMeans(n_clusters=4,random_state=30)
# k_model.fit(course_dig_scaled)
#
# course_dig['target'] = k_model.predict(course_dig_scaled)
#
# sns.scatterplot(data=course_dig,x='mean',y='max',hue='target')
#
# curses_target = course_dig.groupby('target')['课程代码'].agg(list).reset_index()
# print(curses_target)
#
# plt.show()
