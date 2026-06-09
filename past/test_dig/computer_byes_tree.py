import pandas as pd
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import numpy as np

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


label = LabelEncoder()
df['buys_computer'] = label.fit_transform(df['buys_computer'])

# for i in list(df.columns):
#     df[i] = label.fit_transform(df[i])

# df['age'] = label.fit_transform(df['age'])
# df['income'] = label.fit_transform(df['income'])
# df['student'] = label.fit_transform(df['student'])
# df['credit_rating'] = label.fit_transform(df['credit_rating'])
# df['buys_computer'] = label.fit_transform(df['buys_computer'])

one = OneHotEncoder(sparse_output=False)

hot_df = one.fit_transform(df[['age', 'income', 'student', 'credit_rating']])

#print(df)
print('************')

x = hot_df
y = df.iloc[:,-1]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)

gas = GaussianNB()

gas.fit(x_train,y_train)


print('***************')

mul = MultinomialNB()
mul.fit(x_train,y_train)


print('***************')
bor = BernoulliNB()
bor.fit(x_train,y_train)




new_data = pd.DataFrame({
    'age': ['<=30'],
    'income': ['medium'],
    'student': ['yes'],
    'credit_rating': ['fair']
})

hot_new_data = one.transform(new_data)

# 高斯模型预测
gaus_pred = gas.predict(hot_new_data)
print(f"高斯朴素贝叶斯预测结果: buys_computer = {label.inverse_transform(gaus_pred)}")

# 多项式模型预测
mul_pred = mul.predict(hot_new_data)
print(f"多项式朴素贝叶斯预测结果: buys_computer = {label.inverse_transform(mul_pred)[0]}")

# 伯努利模型预测
ber_pred = bor.predict(hot_new_data)
print(f"伯努利朴素贝叶斯预测结果: buys_computer = {label.inverse_transform(ber_pred)[0]}")