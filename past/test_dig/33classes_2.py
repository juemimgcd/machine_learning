from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(10,10))

# （3）对“33门课程的考试成绩”进行聚类分析，分析的角度分别从课程和学生的两个维度出发
# ，并对分析结果进行解读，说明使用聚类可以达到何种的决策支持目标

df = pd.read_csv('33classes_scores.csv', encoding='gbk')

data = df[['学号', '课程代码', '期末成绩']]
data = data.fillna(value=0)

student = data.pivot(index='学号',columns='课程代码',values='期末成绩')

student_cluster = student.copy()

student_cluster['avg'] = student.mean(axis=1)
student_cluster['min'] = student.min(axis=1)

sc = StandardScaler()

student_scaled = sc.fit_transform(student_cluster[['avg','min']])

k_model = KMeans(n_clusters=4,random_state=30)

student_cluster['target'] = k_model.fit_predict(student_scaled)

sns.scatterplot(data=student_cluster,x='avg',y='min',hue='target')
plt.show()