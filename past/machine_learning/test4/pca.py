import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance

# --- 1. 导入数据并预处理 ---
train_df = pd.read_csv('train.csv')
X = train_df.drop(columns=['price_range'])
y = train_df['price_range']

# 添加一些衍生特征
X['screen_area'] = X['sc_h'] * X['sc_w']
X['pixel_total'] = X['px_height'] * X['px_width']

# 切分数据
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 标准化（逻辑回归需要）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# --- 2. 特征重要性分析 ---

# 为了可视化定义函数
def plot_top_features(importance_dict, title, top_n=10):
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_items = [item for item in sorted_items if not pd.isna(item[1])] # 过滤NaN
    sorted_items = sorted_items[:top_n]
    features, scores = zip(*sorted_items)
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.barh(range(len(features)), scores)
    plt.yticks(range(len(features)), features)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# --- 方法一：相关性分析 ---
print("--- 方法一：相关性分析 ---")
corrs = X_train.corrwith(y_train).abs().sort_values(ascending=False)
correlation_importance = corrs.to_dict()
print(corrs.head(10))
plot_top_features(correlation_importance, "Method 1: Correlation with Target", 10)

# --- 方法二：随机森林重要性 ---
print("\n--- 方法二：随机森林重要性 ---")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_importance = dict(zip(X_train.columns, rf.feature_importances_))
print(pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(10))
plot_top_features(rf_importance, "Method 2: Random Forest Importance", 10)

# --- 方法三：排列重要性 ---
print("\n--- 方法三：排列重要性 ---")
perm_result = permutation_importance(rf, X_val, y_val, n_repeats=5, random_state=42)
perm_importance = dict(zip(X_val.columns, perm_result.importances_mean))
print(pd.Series(perm_result.importances_mean, index=X_val.columns).sort_values(ascending=False).head(10))
plot_top_features(perm_importance, "Method 3: Permutation Importance", 10)

# --- 方法四：逻辑回归系数分析 ---
print("\n--- 方法四：逻辑回归系数分析 ---")
lr = LogisticRegression(multi_class='ovr', max_iter=1000, random_state=42) # ovr: one-vs-rest
lr.fit(X_train_scaled, y_train)

# 对于多分类，取各特征在所有类别中系数绝对值的最大值作为重要性
coef_abs_max = np.max(np.abs(lr.coef_), axis=0)
lr_importance = dict(zip(X_train.columns, coef_abs_max))
print(pd.Series(coef_abs_max, index=X_train.columns).sort_values(ascending=False).head(10))
plot_top_features(lr_importance, "Method 4: Logistic Regression Coefficients", 10)

print("\n--- 分析完成 ---")