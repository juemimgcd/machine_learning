import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
plt.figure(figsize=(8,5))

iris = load_iris()

data = iris['data']

target = iris['target']

print(iris['feature_names'])

df = pd.DataFrame(
    data=np.c_[data,target],
    columns=list(iris['feature_names']) +['target']
)

print(df.head())
ax1 = plt.subplot(121)
sns.stripplot(x='target',y='sepal length (cm)',data=df,jitter=True,hue='target')
ax2 = plt.subplot(122)
sns.boxplot(x='target',y='sepal length (cm)',data=df,hue='target')
plt.show()