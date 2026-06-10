import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler,FunctionTransformer


def sigmod(x):
    x_clipped = np.clip(x, -100, 100)
    return 1 / (1 + np.exp(-x_clipped))


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


mas = MinMaxScaler()

sc = StandardScaler()

sig = FunctionTransformer(sigmod)


print('************')
new_df1 = mas.fit_transform(df)
print(new_df1)


print('*************')
new_df2 = sc.fit_transform(df)
print(new_df2)


print('**************')

print('************')
new_df3 = sig.fit_transform(df)

print(new_df3)
