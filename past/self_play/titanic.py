import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(12,10))

imp1 = SimpleImputer(missing_values=np.nan,strategy="mean")
encoder = LabelEncoder()

df= pd.read_csv('ths/Titanic_train.csv')

df.drop(['Name','Cabin','Ticket'],axis=1,inplace=True)
print(df.info())
df[['Age']] = imp1.fit_transform(df[['Age']])

df.dropna(axis=0,inplace=True)



df['Sex'] = encoder.fit_transform(df['Sex'])
df['Embarked'] = encoder.fit_transform(df['Embarked'])
print(df.head())
x = df.iloc[:,df.columns != "Survived"]
y = df.iloc[:,1]

scores = []
#n_estimators

for i in range(100,200,10):
     model = RandomForestClassifier(random_state=40,n_estimators=i)
     score = cross_val_score(model,x,y,cv=5).mean()
     scores.append(score)

plt.plot(range(1,11),scores)
plt.show()
# print(max(scores))
# n_estimators = 190
#0.812

#max_depth=4
# 0.811
# param_dep = {'max_depth':np.arange(2,5)}
#
# model = RandomForestClassifier(random_state=50,n_estimators=190)
# gs = GridSearchCV(model,param_dep,cv=5)
# gs.fit(x,y)
# print(gs.best_score_)
# print(gs.best_params_)

