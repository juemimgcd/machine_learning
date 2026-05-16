from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

iris = load_iris()

data = iris['data']

target = iris['target']

print(iris['feature_names'])

df = pd.DataFrame(
    data=data,
    columns=iris['feature_names']
)
print(df)

data_train,data_test,target_train,target_test = train_test_split(data,target,test_size=0.3)

lg = LogisticRegression(max_iter=100)

lg.fit(data_train,target_train)
print(lg.score(data_train, target_train))

print(lg.score(data_test, target_test))

target_pred = lg.predict(data_test)

print(confusion_matrix(target_test, target_pred))