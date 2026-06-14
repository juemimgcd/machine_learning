from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(8,8))

# x = np.array([[2],[3],[4],[5],[6]])
# y = np.array([[2.2],[3.8],[5.5],[6.5],[7.0]])
#
#
l_model = LinearRegression()
#
# l_model.fit(x,y)

# print(l_model.predict(np.array([[10]]))[0])


x = np.array([115, 110, 80, 135, 105]).reshape(-1, 1)
y = np.array([24.8, 21.6, 18.4, 29.2, 22])

l_model.fit(x,y)

y2_pre = l_model.predict([[110]])

R = y[1] - y2_pre
print(R)

plt.scatter(x,y)
plt.plot(x,l_model.predict(x),c='r')
plt.show()