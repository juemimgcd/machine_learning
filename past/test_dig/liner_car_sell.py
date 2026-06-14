import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.array([2, 3, 4, 5, 6]).reshape(-1, 1)
y = [2.2, 3.8, 5.5, 6.5, 7.0]

lg = LinearRegression()

lg.fit(x, y)
x_test = np.array([10]).reshape(-1,1)
y_pre = lg.predict(x_test)

print(f'第十年的费用是{y_pre[0]}')

print(f'y = {lg.coef_[0]}x + {lg.intercept_}')

x_test2 = np.array([3]).reshape(-1,1)
y2_pre = lg.predict(x_test2)

R = y[1] - y2_pre

print(f'第二年费用的残差为{R[0]}')
