import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,f1_score

digits = load_digits()
# print(digits['feature_names'])
print(digits['data'])
x,y = digits['data'],digits['target']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=30)

gn = GaussianNB()

gn.fit(x_train,y_train)

print(gn.score(x_test, y_test))

y_pred = gn.predict(x_test)

matrix = confusion_matrix(y_test,y_pred)
f1_score  = f1_score(y_test,y_pred,average='weighted')
print(f1_score)
print(matrix)