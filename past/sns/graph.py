import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(12,10))

arr = np.random.rand(100)

sns.displot(arr,bins=10,kde=True,rug=True)
plt.show()