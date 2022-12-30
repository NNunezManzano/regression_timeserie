import pandas as pd
import numpy as np
import os

import sklearn as sk
from sklearn import metrics
import lightgbm as lgb


# set workdir
os.chdir("/home/nnm/Documents/Proyectos/repos/regression_timeserie")

ticker = "SPY"

data_fe = pd.read_csv(f"./dataset/fe_data_{ticker}.csv")


train = data_fe.iloc[:2718]
valid = data_fe.iloc[2718:2720]

X_train = train.drop(columns=["Close", "Date"])

y_train = train.Close

X_test = valid.drop(columns=["Close", "Date"])

y_test = valid.Close

parameters = {
    "boosting_type": "gbdt",
    "objetive": "regression",
    "metric": "auc",
    "num_leaves": 10,
    "learning_rate": 0.01,
    "num_iterations": 500,
    "seed": 42,
}
gbdt = lgb.LGBMRegressor(
    boosting_type="gbdt",
    objetive="regression",
    metric="auc",
    num_leaves=10,
    learning_rate=0.01,
    num_iterations=5000,
    seed=42,
)

gbdt.fit(X_train, y_train)

y_pred = gbdt.predict(X_test)

print(f"Test score: {metrics.mean_absolute_error(y_pred, y_test)}")
