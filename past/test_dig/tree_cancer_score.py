from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(8, 8))

cancer = load_breast_cancer()

data = cancer['data']
target = cancer['target']

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=50)

dt = DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=4)

dt.fit(x_train, y_train)

y_pre = dt.predict_proba(x_test)[:,1]
print(y_pre)

# pre, rec, th = precision_recall_curve(y_test, y_pre)
#
# plt.plot(pre, rec)
# plt.show()

# f1_scores = cross_val_score(dt,data,target,cv=10,scoring='f1')
# precisions = cross_val_score(dt,data,target,cv=10,scoring='precision')
# recalls = cross_val_score(dt,data,target,cv=10,scoring='recall')
#
# print(f1_scores)
# print(precisions)
# print(recalls)
#
# plt.plot(range(1,11),precisions,label='pre')
# plt.plot(range(1,11),recalls,label='rec')
# plt.legend(loc = 4)
#
# plt.show()
