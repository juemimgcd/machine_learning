# （1）通过breast_cancer数据集，利用已有的贝叶斯分类或决策树分类方法，实现精准率P、召回率R的分析，
# 查看是否存在P，R指标的不可调和的矛盾；
# （2）在iris数据集上，验证是否存在P，R指标的不可调和的矛盾；
# （3）分别使用划分测试集和K折交叉检验的方法，分析和比较breast_cancer数据在两种不同的性能评估方法下
#   其对应的Fscore是否有了优化。
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix as CM,precision_score as P,recall_score as R,precision_recall_curve,f1_score
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(8,8))

cancer = load_breast_cancer()

data = cancer['data']
target = cancer['target']

x_train,x_test,y_train,y_test = train_test_split(data,target,test_size=0.3,random_state=50)

dt = DecisionTreeClassifier(criterion='entropy',splitter='random',max_depth=4)

dt.fit(x_train,y_train)

y_pre = dt.predict(x_test)

percision,recall,this = precision_recall_curve(y_test,y_pre)


plt.plot(percision,recall)
plt.show()








