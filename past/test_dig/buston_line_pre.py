import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# （2）利用线性回归函数实现波士顿房价的预测，并对程序中的randomstate进行调参，
# 对比性能指标的F-SCORE说明randomstate对score是否有影响；
# （3）查阅工具书，明确train_test_spilt函数用处的是什么？调用格式是什么？

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(12,10))

df = pd.read_csv('buston.csv')

print(df.head())
x = df.iloc[:,0:-1]
y = df.iloc[:,-1]

scores = []

for i in range(50,100,10):
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=i)
    lg = LinearRegression()
    lg.fit(x_train,y_train)
    f_score = lg.score(x_test,y_test)
    scores.append(f_score)

# plt.plot(range(50,100,10),scores)
# plt.xlabel('seed')
# plt.xticks(range(50,120))
# plt.ylabel('score')
# plt.title('learning')
# plt.show()

index = np.argmax(scores)

random_seed = index*10 + 50
print(random_seed)