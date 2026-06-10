import numpy as np
import pandas as pd



path = r"/past/test_dig/data.xls"


# 数据按行存储为列表的列表
data = [
    [78, 521, 602, 2863],
    [144, -600, -521, 2245],
    [95, -457, 468, -1283],
    [69, 596, 695, 1054],
    [190, 527, 691, 2051],
    [101, 403, 470, 2487],
    [146, 413, 435, 2571]
]


df = pd.DataFrame(data, columns=["Col1", "Col2", "Col3", "Col4"])
print(df)



def min_max(x):
    return (x - x.min()) / (x.max() - x.min())


def sigmod(x):
    x_clipped = np.clip(x, -100, 100)  # 裁剪到合理范围
    return 1 / (1 + np.exp(-x_clipped))


def z_score(x):
    return (x - x.mean()) / x.std()


df2 = df.apply(min_max)
df3 = df.apply(sigmod)
df4 = df.transform(z_score,axis=0)
print(df4)