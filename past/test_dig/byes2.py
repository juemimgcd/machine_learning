import numpy as np
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB

zero = np.array([[0, 0, 0]])
neg = np.array([[-18, -11, -20]])
mix = np.array([[-1, 0, 1]])

# 简单训练数据
X_train = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
y_train = np.array([0, 1, 0])

gas = GaussianNB()
gas.fit(X_train,y_train)
print(gas.predict_proba(zero))
print(gas.predict_proba(neg))
print(gas.predict_proba(mix))
print('***************')

mul = MultinomialNB()
mul.fit(X_train,y_train)
print(mul.predict_proba(zero))
try:
    print('负值',mul.predict_proba(neg))
except Exception as e:
    print("报错",e)
print(mul.predict_proba(mix))
print('***************')

bor = BernoulliNB()
bor.fit(X_train,y_train)
print(bor.predict_proba(zero))
print(bor.predict_proba(neg))
print(bor.predict_proba(mix))