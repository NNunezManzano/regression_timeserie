"""This script will build the model as is, getting data with ETL, making feature_engineering and running the algorithm"""


import pandas as pd
import numpy as np
import os

import lightgbm as lgb

from data import Data_extract

from multistep import Multistep_reg

# Parameters
ticker = "SPY"

start_date = "20200101"

end_date = "20221231"

test_period = 30


# Data extraction
dex = Data_extract()

data = dex.get_data(ticker, start_date=start_date, end_date=end_date)

# train test split
train = data.iloc[: len(data) - test_period]
valid = data.iloc[len(data) - test_period :]

X_train = train.drop(columns=["Close", "Date"])

y_train = train.Close

X_test = valid.drop(columns=["Close", "Date"])

y_test = valid.Close

# Regresor
gbdt = lgb.LGBMRegressor(
    boosting_type="gbdt",
    objetive="regression",
    metric="auc",
    num_leaves=10,
    learning_rate=0.01,
    num_iterations=5000,
    seed=42,
)

# Multistep regression
msr = Multistep_reg()

msr.lightgb_ms(
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    regresor=gbdt,
    period=test_period,
)
