import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载Wine数据集
wine = load_wine()
X, y = wine.data, wine.target



# 2. 数据分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. 使用OvR策略的逻辑回归模型
# 方法1: 直接使用LogisticRegression，默认multi_class='ovr'
lr_ovr = LogisticRegression(random_state=42, max_iter=1000)



# 训练模型
lr_ovr.fit(X_train_scaled, y_train)

# 预测
y_pred = lr_ovr.predict(X_test_scaled)
y_pred_proba = lr_ovr.predict_proba(X_test_scaled)  # 获取各类别概率

# 5. 模型评估
accuracy = accuracy_score(y_test, y_pred)
print(f"\n模型准确率: {accuracy:.4f}")

print("\n详细分类报告:")
print(classification_report(y_test, y_pred, target_names=wine.target_names))

# 6. 混淆矩阵可视化
plt.figure(figsize=(12, 4))

# 子图1: 混淆矩阵
plt.subplot(1, 3, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=wine.target_names,
            yticklabels=wine.target_names)
plt.title('混淆矩阵')
plt.xlabel('预测标签')
plt.ylabel('真实标签')

# 子图2: 各类别预测概率分布
plt.subplot(1, 3, 2)
for i in range(len(wine.target_names)):
    plt.hist(y_pred_proba[:, i], alpha=0.6, label=f'{wine.target_names[i]}', bins=15)
plt.xlabel('预测概率')
plt.ylabel('样本数量')
plt.title('各类型酒的预测概率分布')
plt.legend()

# 子图3: 特征重要性（系数绝对值）
plt.subplot(1, 3, 3)
feature_importance = np.abs(lr_ovr.coef_[0])  # 取第一个类别的系数作为示例
top_features_idx = np.argsort(feature_importance)[-10:]  # 取前10个重要特征
top_features_names = [wine.feature_names[i] for i in top_features_idx]
top_features_values = feature_importance[top_features_idx]

plt.barh(range(len(top_features_names)), top_features_values)
plt.yticks(range(len(top_features_names)), top_features_names)
plt.xlabel('系数绝对值')
plt.title('Top 10 重要特征')
plt.tight_layout()
plt.show()

# 7. OvR策略解释
print("\nOvR策略工作原理:")
print("- OvR为每个类别训练一个二分类器")
print("- 每个分类器区分一个特定类别与其他所有类别")
print("- 在预测时，选择具有最高置信度得分的类别作为预测结果")
print(f"- Wine数据集有{len(np.unique(y))}个类别，因此训练了{len(np.unique(y))}个二分类器")

# 8. 查看各个二分类器的信息
print(f"\n各二分类器的系数形状: {lr_ovr.coef_.shape}")
print("(类别数, 特征数) -> 每行代表一个二分类器的系数向量")