from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(12,10))

cancer = load_breast_cancer()

data = cancer['data']
target = cancer['target']

df = pd.DataFrame(
    data=np.c_[data,target],
    columns=list(cancer['feature_names']) + ['target']

)

print(df.head())
sc = StandardScaler()
data_sc = sc.fit_transform(data)

x_train,x_test,y_train,y_test = train_test_split(data_sc,target,test_size=0.3,random_state=15)

lr = LogisticRegression(max_iter=500)

lr.fit(x_train,y_train)

print(lr.score(x_test, y_test))

print(lr.predict(x_test[2:5]))