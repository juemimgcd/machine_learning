import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, accuracy_score)

# 1. 加载数据
wine = load_wine()
X = wine.data
y = wine.target
target_names = wine.target_names

# 2. 划分训练集和测试集 (7:3 分割)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 数据标准化 (逻辑回归和随机森林对尺度敏感，建议标准化)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"数据形状: {X.shape}, 训练集: {X_train.shape}, 测试集: {X_test.shape}")


from sklearn.tree import DecisionTreeClassifier

# 定义模型
dt_gini = DecisionTreeClassifier(criterion='gini', random_state=42)
dt_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)

# 训练模型
dt_gini.fit(X_train, y_train)
dt_entropy.fit(X_train, y_train)

# 预测
y_pred_gini = dt_gini.predict(X_test)
y_pred_entropy = dt_entropy.predict(X_test)

# 评估
print("=== 决策树：Gini系数 ===")
print(f"准确率: {accuracy_score(y_test, y_pred_gini):.4f}")
print(classification_report(y_test, y_pred_gini, target_names=target_names))

print("\n=== 决策树：信息增益 (Entropy) ===")
print(f"准确率: {accuracy_score(y_test, y_pred_entropy):.4f}")
print(classification_report(y_test, y_pred_entropy, target_names=target_names))



