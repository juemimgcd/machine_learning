import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, accuracy_score)

# 设置绘图风格
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

# 1. 加载数据
wine = load_wine()
X, y = wine.data, wine.target
target_names = wine.target_names

# 2. 划分训练集和测试集 (7:3分割，保持类别比例一致)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. 特征标准化 (逻辑回归和SVM必须，树模型虽不敏感但也建议统一处理流程)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 定义模型
models = {
    "逻辑回归": LogisticRegression(max_iter=1000, random_state=42),
    "决策树": DecisionTreeClassifier(random_state=42),
    "随机森林": RandomForestClassifier(n_estimators=100, random_state=42)
}

# 存储预测结果用于后续绘图
y_pred_dict = {}

print("--- 模型性能对比 ---\n")

plt.figure(figsize=(20, 5))

for i, (name, model) in enumerate(models.items()):
    # 训练模型
    # 注意：逻辑回归使用标准化数据，树模型其实可以用原始数据，但这里统一用标准化数据
    model.fit(X_train_scaled, y_train)

    # 预测
    y_pred = model.predict(X_test_scaled)
    y_pred_dict[name] = y_pred

    # 计算准确率
    acc = accuracy_score(y_test, y_pred)

    # 打印分类报告
    print(f"【{name}】 准确率: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # 绘制混淆矩阵
    plt.subplot(1, 3, i+1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'{name} 混淆矩阵')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')

plt.tight_layout()
plt.show()







