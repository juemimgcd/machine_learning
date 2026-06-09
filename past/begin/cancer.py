from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


cancer = load_breast_cancer()

data = cancer['data']
target = cancer['target']

data_train,data_test,target_train,target_test = train_test_split(data,target,test_size=0.3,random_state=420)

#model = KNeighborsClassifier(n_neighbors=4)

#model.fit(data_train,target_train)

#score = model.score(data_test,target_test)
#print(score)

score = []

k_range = range(1,20)

for i in k_range:
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(data_train,target_train)
    score.append(model.score(data_test,target_test))

plt.plot(k_range,score)
plt.xticks(range(1,20))
plt.title('学习曲线')
plt.show()

