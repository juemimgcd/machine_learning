import pandas as pd

import xgboost as xgb

df = pd.DataFrame({'x':[1,2,3], 'y':[10,20,30]})

X_train = df.drop('y',axis=1)

Y_train = df['y']

T_train_xgb = xgb.DMatrix(X_train,Y_train)

params = {"objective": "reg:linear", "booster":"gblinear"}

gbm = xgb.train(dtrain=T_train_xgb,params=params)

Y_pred = gbm.predict(xgb.DMatrix(pd.DataFrame({'x':[4,5]})))

print(Y_pred)