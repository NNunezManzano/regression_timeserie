""" This script will run if multistep was set on model.py """

import os

import pandas as pd

from feature_engineering import Feature_engineering as fe

try:
    os.chdir("../regression_timeserie")
except:
    os.chdir("../")


col = {
    "col1": ["volume", "guru", "guru", "guru", "guru", "guru"],
    "col2": [1, 2, 3, 4, 5, 6],
}

df = pd.DataFrame.from_dict(col)

df = fe.lags(df, 2, "col2")


print(df)
