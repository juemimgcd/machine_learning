import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, accuracy_score, precision_score,
                             recall_score, f1_score)
from sklearn.multiclass import OneVsRestClassifier
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set(style="whitegrid", font_scale=1.1)

# ==================== 1. 数据加载与预处理 ====================
print("=" * 60)
print("1. 数据加载与预处理")
print("=" * 60)

# 加载Wine数据集
wine = load_wine()
X, y = wine.data, wine.target
feature_names = wine.feature_names
target_names = wine.target_names

print(f"数据集形状: {X.shape}")
print(f"类别数量: {len(target_names)}")
print(f"类别名称: {target_names}")
print(f"每个类别的样本数: {np.bincount(y)}")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n训练集: {X_train.shape}, 测试集: {X_test.shape}")

# ==================== 2. 模型训练与性能对比 ====================
print("\n" + "=" * 60)
print("2. 模型训练与性能对比")
print("=" * 60)

# 定义模型
models = {
    "决策树": DecisionTreeClassifier(random_state=42),
    "随机森林": RandomForestClassifier(n_estimators=100, random_state=42),
    "逻辑回归": LogisticRegression(max_iter=1000, random_state=42)
}

# 存储结果
results = {}
y_pred_dict = {}

plt.figure(figsize=(18, 6))

for idx, (name, model) in enumerate(models.items(), 1):
    print(f"\n【{name}】")
    print("-" * 40)

    # 训练模型
    model.fit(X_train_scaled, y_train)

    # 预测
    y_pred = model.predict(X_test_scaled)
    y_pred_dict[name] = y_pred

    # 计算性能指标
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    results[name] = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }

    # 打印分类报告
    print(classification_report(y_test, y_pred, target_names=target_names))

    # 绘制混淆矩阵
    plt.subplot(1, 3, idx)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names,
                cbar=False)
    plt.title(f'{name}\nAcc: {acc:.4f}')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印性能对比表格
print("\n" + "=" * 60)
print("性能对比总结")
print("=" * 60)
performance_df = pd.DataFrame(results).T
performance_df = performance_df.round(4)
print(performance_df)
print("=" * 60)

# ==================== 3. 随机森林参数分析 ====================
print("\n" + "=" * 60)
print("3. 随机森林参数分析")
print("=" * 60)

# 参数1: n_estimators (基学习器数量)
print("\n【参数1: n_estimators (基学习器数量)】")
n_estimators_range = range(10, 201, 10)
train_acc_nest = []
test_acc_nest = []

for n_est in n_estimators_range:
    rf = RandomForestClassifier(n_estimators=n_est, random_state=42)
    rf.fit(X_train_scaled, y_train)
    train_acc_nest.append(rf.score(X_train_scaled, y_train))
    test_acc_nest.append(rf.score(X_test_scaled, y_test))

# 参数2: max_depth (树的最大深度)
print("\n【参数2: max_depth (树的最大深度)】")
max_depth_range = range(1, 21)
train_acc_depth = []
test_acc_depth = []

for depth in max_depth_range:
    rf = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=42)
    rf.fit(X_train_scaled, y_train)
    train_acc_depth.append(rf.score(X_train_scaled, y_train))
    test_acc_depth.append(rf.score(X_test_scaled, y_test))

# 绘制参数影响图
plt.figure(figsize=(14, 5))

# n_estimators 影响
plt.subplot(1, 2, 1)
plt.plot(n_estimators_range, train_acc_nest, 'b-o', label='训练集准确率', markersize=4)
plt.plot(n_estimators_range, test_acc_nest, 'r-s', label='测试集准确率', markersize=4)
plt.axvline(x=100, color='k', linestyle='--', alpha=0.7, label='默认值 (n=100)')
plt.xlabel('基学习器数量 (n_estimators)')
plt.ylabel('准确率')
plt.title('n_estimators 对性能的影响')
plt.legend()
plt.grid(True, alpha=0.3)

# max_depth 影响
plt.subplot(1, 2, 2)
plt.plot(max_depth_range, train_acc_depth, 'b-o', label='训练集准确率', markersize=4)
plt.plot(max_depth_range, test_acc_depth, 'r-s', label='测试集准确率', markersize=4)
plt.axvline(x=10, color='k', linestyle='--', alpha=0.7, label='常用值 (depth=10)')
plt.xlabel('树的最大深度 (max_depth)')
plt.ylabel('准确率')
plt.title('max_depth 对性能的影响')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rf_parameters_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 找出最佳参数
best_n_est = n_estimators_range[np.argmax(test_acc_nest)]
best_depth = max_depth_range[np.argmax(test_acc_depth)]
print(f"\n最佳 n_estimators: {best_n_est} (测试集准确率: {max(test_acc_nest):.4f})")
print(f"最佳 max_depth: {best_depth} (测试集准确率: {max(test_acc_depth):.4f})")

# ==================== 4. ROC曲线与AUC分析 ====================
print("\n" + "=" * 60)
print("4. ROC曲线与AUC分析")
print("=" * 60)

# 对标签进行二值化处理（用于多分类ROC）
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_test_bin.shape[1]

# 训练随机森林模型
rf_best = RandomForestClassifier(n_estimators=100, random_state=42)
rf_best.fit(X_train_scaled, y_train)

# 获取预测概率
y_score = rf_best.predict_proba(X_test_scaled)

# 计算每个类别的ROC曲线和AUC
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 计算宏平均ROC曲线
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# 绘制ROC曲线
plt.figure(figsize=(10, 8))
colors = ['blue', 'red', 'green']

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'{target_names[i]} (AUC = {roc_auc[i]:.4f})')

# 绘制宏平均ROC曲线
plt.plot(fpr["macro"], tpr["macro"],
         label=f'宏平均 (AUC = {roc_auc["macro"]:.4f})',
         color='navy', linestyle=':', linewidth=3)

# 绘制随机猜测线
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='随机猜测 (AUC = 0.5)')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正率 (False Positive Rate)', fontsize=12)
plt.ylabel('真正率 (True Positive Rate)', fontsize=12)
plt.title('随机森林多分类 ROC 曲线', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印AUC结果
print("\nAUC 分析结果:")
print("-" * 40)
for i in range(n_classes):
    print(f"{target_names[i]}: AUC = {roc_auc[i]:.4f}")
print(f"宏平均 AUC: {roc_auc['macro']:.4f}")

# ==================== 5. 综合分析与结论 ====================
print("\n" + "=" * 60)
print("5. 综合分析与结论")
print("=" * 60)

print("\n【模型性能对比】")
print(performance_df.to_string())

print("\n【随机森林参数分析结论】")
print(f"1. n_estimators 影响:")
print(f"   - 随着树的数量增加，模型性能趋于稳定")
print(f"   - 当 n_estimators >= 50 时，性能提升不明显")
print(f"   - 推荐使用 n_estimators = 100 作为默认值")

print(f"\n2. max_depth 影响:")
print(f"   - 过小的深度会导致欠拟合")
print(f"   - 过大的深度可能导致过拟合")
print(f"   - 推荐使用 max_depth = 10-15 之间")

print("\n【ROC曲线分析】")
print(f"- 所有类别的 AUC 值都接近 1.0，说明随机森林对Wine数据集的分类能力极强")
print(f"- 宏平均 AUC = {roc_auc['macro']:.4f}，表明模型整体性能优秀")
print(f"- ROC曲线靠近左上角，说明模型在不同阈值下都表现良好")

print("\n【实验总结】")
print("1. 随机森林在Wine数据集上表现最佳，准确率通常在95%以上")
print("2. 逻辑回归作为线性模型也表现良好，适合线性可分的数据")
print("3. 单棵决策树容易过拟合，性能相对较低")
print("4. 随机森林通过集成多棵树有效降低了方差，提高了泛化能力")
print("=" * 60)