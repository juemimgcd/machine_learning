import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# 1. 数据准备（同上，用 DataFrame）
data = pd.DataFrame({
    '房屋面积': [115, 110, 80, 135, 105],
    '销售价格': [24.8, 21.6, 18.4, 29.2, 22]
})
x = np.array([115, 110, 80, 135, 105]).reshape(-1, 1)  # 房屋面积
y = np.array([24.8, 21.6, 18.4, 29.2, 22])
model = LinearRegression()
model.fit(x,y)
# coefficient = model.coef_[0]       # 系数
# intercept = model.intercept_# 截距
# x_pre = np.array([150]).reshape(1,-1)
# y_pred_150 = model.predict(x_pre)[0]  # 150m²售价预测
#
# # 3. 输出题目答案（保留3位小数）
# print(f"(1) 系数: {coefficient:.3f}")
# print(f"(2) 截距: {intercept:.3f}")
# print(f"(3) 150m²售价预测: {y_pred_150:.3f} 万元\n")
y_pred = model.predict(x)

# 计算残差
residuals = y - y_pred

# 计算残差平方和（RSS）
rss = np.sum(residuals ** 2)

# 计算均方误差（MSE）
mse = rss / len(y)

print("残差平方和（RSS）:", rss)
print("均方误差（MSE）:", mse)



# # 2. 用 jointplot 绘制（同时显示散点、回归线、边际分布）
# sns.jointplot(
#     data=data,
#     x='房屋面积',
#     y='销售价格',
#     kind='reg',  # 指定类型为“回归”（自动绘制散点+拟合线）
#     scatter_kws={'color': 'teal', 's': 60},  # 散点颜色（青绿色）
#     line_kws={'color': 'orange', 'lw': 2},  # 拟合线颜色（橙色）
#     height=6,
#     marginal_ticks=True  # 边际图显示刻度
# )
#
# # 添加标题
# plt.suptitle('房屋面积与销售价格的关系（含边际分布）', y=1.02, fontsize=14)
# plt.show()