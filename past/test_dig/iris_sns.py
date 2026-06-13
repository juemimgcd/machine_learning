from pandas.plotting import parallel_coordinates
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(12,10))


iris = load_iris()

data = iris['data']
target = iris['target']
print(iris['feature_names'])
ful_data = np.c_[data,target]

df = pd.DataFrame(
    data=ful_data,
    columns=list(iris['feature_names']) +['target']
)

ax1 = plt.subplot(221)
sns.stripplot(x='target',y='sepal length (cm)',data=df,jitter=True,hue='target')
ax2 = plt.subplot(222)
sns.stripplot(x='target',y='sepal width (cm)',data=df,jitter=True,hue='target')
ax3 = plt.subplot(223)
plt.boxplot([df.iloc[:,0],df.iloc[:,1],df.iloc[:,2],df.iloc[:,3]],labels=['sepal length', 'sepal width', 'petal length', 'petal width'])

ax4 = plt.subplot(224)
sns.violinplot([df.iloc[:,0],df.iloc[:,1],df.iloc[:,2],df.iloc[:,3]])
plt.show()

plt.figure(figsize=(12, 6))
parallel_coordinates(df, 'target', color=['#FF0000', '#00FF00', '#0000FF'])
plt.title('鸢尾花数据集平行坐标图')
plt.ylabel('测量值(cm)')
plt.grid(alpha=0.3)
plt.show()