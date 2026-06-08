import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score,train_test_split,GridSearchCV
import matplotlib.pyplot as plt
import graphviz
from sklearn.tree import export_graphviz

df = pd.read_csv('data/buston.csv')
# print(df.info())

df.dropna(inplace=True)


print(df.head())
sc = StandardScaler()

x = df.iloc[:,0:-1]
y = df.iloc[:,-1]
x_scaled = sc.fit_transform(x)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=20)


print(df.info())


param = {
    'n_estimators':[100,130,150,170,190],
    'max_depth':[3,5,7]
}
model = RandomForestRegressor()

gs = GridSearchCV(model,param,cv=4)

gs.fit(x_train,y_train)

print(gs.best_params_)
print(gs.best_score_)

best_forest = gs.best_estimator_

# {'max_depth': 7, 'n_estimators': 150}
# 0.8204016487096331

tree = best_forest[3]



dot = export_graphviz(tree,
                           feature_names=x.columns,
                           class_names=['yes','no'],
                           filled=True,
                           rounded=True,
                           fontname='SimHei'
                           )

graph = graphviz.Source(dot.encode('utf-8').decode('utf-8'), format='png', filename='boston_tree')

graph.view('boston_tree')