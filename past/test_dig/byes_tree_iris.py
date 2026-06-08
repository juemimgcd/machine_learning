from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np


iris = load_iris()
data = iris['data']
target = iris['target']


x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=50)

depths = np.arange(1, 20)
precisions = []
recalls = []

for depth in depths:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=50)
    dt.fit(x_train, y_train)
    y_pred = dt.predict(x_test)

    p = precision_score(y_test, y_pred, average='weighted')
    r = recall_score(y_test, y_pred, average='weighted')

    precisions.append(p)
    recalls.append(r)

# 绘制结果
plt.figure(figsize=(10, 5))
plt.plot(depths, precisions, 'b-', label='Precision')
plt.plot(depths, recalls, 'r-', label='Recall')
plt.xlabel('Tree Depth')
plt.ylabel('Score')
plt.legend()
plt.grid()
plt.show()








