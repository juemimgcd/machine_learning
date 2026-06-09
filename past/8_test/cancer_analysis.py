from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,BaggingClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(8,8))

cancer = load_breast_cancer()
data = cancer['data']
target = cancer['target']
x_train,x_test,y_train,y_test = train_test_split(data,target,test_size=0.3,random_state=30)

# 使用决策树CART
model1 = DecisionTreeClassifier(max_depth=4,random_state=42)
model1.fit(x_train,y_train)
y1_pre = model1.predict(x_test)
cart_f_score = f1_score(y_test,y1_pre)
cart_test_score = cross_val_score(model1,data,target,cv=10).mean()

# 贝叶斯分类
model2 = GaussianNB()
model2.fit(x_train,y_train)
y2_pre = model2.predict(x_test)
nb_f_score = f1_score(y_test,y2_pre)
nb_test_score = cross_val_score(model2,data,target,cv=10).mean()


# 逻辑回归
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_train)
model3 = LogisticRegression(solver='sag', random_state=42, max_iter=5000, tol=1e-4)
model3.fit(x_scaled, y_train)
x_test_scaled = scaler.transform(x_test)
y3_pre = model3.predict(x_test_scaled)
lg_f_score = f1_score(y_test,y3_pre)
lg_test_score = cross_val_score(model3, scaler.transform(data), target, cv=10).mean()


#随机森林RF
model4 = RandomForestClassifier(max_depth=4,random_state=42)
model4.fit(x_train,y_train)
y_pre4 = model4.predict(x_test)
rf_f_score = f1_score(y_test,y_pre4)
rf_test_score = cross_val_score(model4,data,target,cv=10).mean()

# bagging
model5 = BaggingClassifier(estimator=DecisionTreeClassifier(),n_estimators=100,random_state=42)
model5.fit(x_train,y_train)
y_pre5 = model5.predict(x_test)
bg_f_score = f1_score(y_test,y_pre5)

bg_test_score = cross_val_score(model5,data,target,cv=10).mean()

# GBDT
model6 = GradientBoostingClassifier(max_depth=4,n_estimators=100,random_state=42)
model6.fit(x_train,y_train)
y_pre6 = model6.predict(x_test)
gb_f_score = f1_score(y_test,y_pre6)

gb_test_score = cross_val_score(model6,data,target,cv=10).mean()

#KNN
model7 = KNeighborsClassifier()
model7.fit(x_train,y_train)
y_pre7 = model7.predict(x_test)
kn_f_score = f1_score(y_test,y_pre7)

kn_test_score = cross_val_score(model7,data,target,cv=10).mean()

f_score = [cart_f_score,nb_f_score,lg_f_score,rf_f_score,bg_f_score,gb_f_score,kn_f_score]
test_score = [cart_test_score,nb_test_score,lg_test_score,rf_test_score,bg_test_score,gb_test_score,kn_test_score]


dist = {
    'name':['cart','nb','lg','rf','bg','gb','knn'],
    'f_score':f_score,
    'test_score':test_score
}
df = pd.DataFrame(dist)
# print(df)

x = np.arange(len(df))
w = 0.3
plt.bar(x - w,f_score,width=0.3,label='f_score',color='red')
plt.bar(x,test_score,width=0.3,label='test',color='black')
plt.xlabel('models')
plt.ylabel('score')
plt.legend(loc=4)
plt.show()