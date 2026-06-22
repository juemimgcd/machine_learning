from sklearn.datasets import load_wine
from sklearn import tree
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import graphviz

wine = load_wine()

data = wine['data']
target = wine['target']
feature_names = wine['feature_names']

df = pd.DataFrame(
    data=np.c_[data,target],
    columns=list(feature_names) +['target']
)
print(df.head())

x_train,x_test,y_train,y_test = train_test_split(data,target,test_size=0.3,random_state=30)

# clf = tree.DecisionTreeClassifier(criterion='entropy',
#                                   random_state=30,
#                                   splitter='random',
#                                   max_depth=4
#                                   )
# clf.fit(x_train,y_train)
#
# print(clf.score(x_test,y_test))

# dot = tree.export_graphviz(clf,
#                            feature_names=feature_names,
#                            class_names=['朗姆酒','苦艾酒','琴酒'],
#                            filled=True,
#                            rounded=True,
#                            fontname='SimHei'
#                            )
#
# graph = graphviz.Source(dot.encode('utf-8').decode('utf-8'),format='png',filename='wine_tree2')
#
# graph.view('wine_tree2')

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# score = []
# for i in range(1,10):
#
#     clf2 = tree.DecisionTreeClassifier(criterion='entropy',
#                                         random_state=30,
#                                         splitter='random',
#                                         max_depth=i+1
#                                         )
#     clf2.fit(x_train,y_train)
#     score.append(clf2.score(x_test,y_test))
#
#
# plt.plot(range(1,10),score)
# plt.show()