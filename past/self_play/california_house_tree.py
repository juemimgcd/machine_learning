from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('ths/housing.csv')

new_df = df.drop(['ocean_proximity'], axis=1)

print(new_df.head())
sc = StandardScaler()

x = new_df.iloc[:, 0:-1]
y = new_df.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=30, test_size=0.3)
score = []

# for i in range(1,20):
#
#     reg = tree.DecisionTreeRegressor(max_depth=i,random_state=0)
#     reg.fit(x_train,y_train)
#     score.append(reg.score(x_test,y_test))

for i in range(2, 6):
    model = RandomForestRegressor(max_depth=4, random_state=50)
    model.fit(x_train, y_train)

    score.append(model.score(x_test, y_test))

plt.plot(range(1, 5), score)
plt.show()
