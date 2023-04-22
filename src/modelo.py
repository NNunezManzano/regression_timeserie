"""This script will build the model as is, getting data with ETL, making feature_engineering and running the algorithm"""


import lightgbm as lgb

from data import Data_extract

from feature_engineering import Feature_engineering

from multistep import Multistep_reg

from sklearn import metrics

# Set Parameters
Params = {
    "dex": {
        "ticker": "SPY",
        "start_date": "20200101",
        "end_date": "20211231",
    },
    "fe": {
        "lags": True,
        "nlags": 60,
        "lags_columns": ["Close"],
        "trend": True,
        "trends": [20, 40],
        "trend_columns": ["Close"],
        "stock": False,
    },
    "test_set": {"test_period": 30, "to_predict": "Close"},
}

# Initialazing etl classes
# Data extraction
dex = Data_extract()

# Feature engenieering

fe = Feature_engineering()


# get data from yfinance
data = dex.get_data(
    Params["dex"]["ticker"],
    start_date=Params["dex"]["start_date"],
    end_date=Params["dex"]["end_date"],
)

# Appling lags
if Params["fe"]["lags"]:
    for column in Params["fe"]["lags_columns"]:
        data = fe.lags(data, n_lags=Params["fe"]["nlags"], column=column)

# train test split
train = data.iloc[: len(data) - Params["test_set"]["test_period"]]
valid = data.iloc[len(data) - Params["test_set"]["test_period"] :]

# Train Test split
X_train = train.drop(columns=[Params["test_set"]["to_predict"], "Date"])

y_train = train[Params["test_set"]["to_predict"]]

X_test = valid.drop(columns=[Params["test_set"]["to_predict"], "Date"])

y_test = valid[Params["test_set"]["to_predict"]]

# Appling trends
if Params["fe"]["trend"]:
    for column in Params["fe"]["trend_columns"]:
        for trend in Params["fe"]["trends"]:
            X_train = fe.trend(X_train, trend=trend, column=column)

if Params["fe"]["stock"]:
    X_train = fe.stocks_ts(X_train)

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

predict = msr.lightgb_ms(
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    regresor=gbdt,
    period=Params["test_set"]["test_period"],
    params=Params,
)


print(metrics.mean_absolute_error(predict, y_test))
