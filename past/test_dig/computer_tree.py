import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import numpy as np
import graphviz


# 定义基础数据
ages = ['<=30', '31...40', '>40']
incomes = ['low', 'medium', 'high']
students = ['no', 'yes']
credit_ratings = ['fair', 'excellent']

# 生成扩充数据
data = []
for _ in range(100):
    age = np.random.choice(ages, p=[0.4, 0.3, 0.3])
    income = np.random.choice(incomes, p=[0.3, 0.4, 0.3])
    student = np.random.choice(students, p=[0.6, 0.4])
    credit = np.random.choice(credit_ratings, p=[0.7, 0.3])

    # 基于业务规则定义购买决策逻辑
    if age == '<=30':
        if student == 'yes' and income != 'low':
            buys = 'yes' if credit == 'excellent' else 'no'
        else:
            buys = 'no'
    elif age == '31...40':
        buys = 'yes' if income != 'low' else 'no'
    else:  # >40
        buys = 'yes' if credit == 'fair' else 'no'

    # 添加10%的随机噪声
    if np.random.random() < 0.1:
        buys = 'yes' if buys == 'no' else 'no'

    data.append([age, income, student, credit, buys])

# 创建DataFrame
columns = ['age', 'income', 'student', 'credit_rating', 'buys_computer']
df = pd.DataFrame(data, columns=columns)

one = OneHotEncoder(sparse_output=False)
hot_df = one.fit_transform(df[['age', 'income', 'student', 'credit_rating']])
feature_names = one.get_feature_names_out(['age', 'income', 'student', 'credit_rating'])

# 用onehot编码就用onehot取出来

x = hot_df
y = df['buys_computer'].map({'yes':1,'no':0})

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)

clf = tree.DecisionTreeClassifier(criterion='entropy',random_state=30,max_depth=4,splitter='random')
clf.fit(x_train,y_train)

dot = tree.export_graphviz(
    clf,
    feature_names=feature_names,
    class_names=['no','yes']
)

graph = graphviz.Source(dot.encode('utf-8').decode('utf-8'), filename='computer_buy', format='png')
graph.view('computer_buy')