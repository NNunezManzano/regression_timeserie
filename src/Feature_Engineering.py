import pandas as pd
import os
import numpy as np

# set workdir
os.chdir("/home/nnm/Documents/Proyectos/repos/regression_timeserie")

ticker = "SPY"

# load data to dataframe
data = pd.read_csv(f"./dataset/data_{ticker}.csv")

#
data["daily_market_cap"] = data.Close * data.Volume

data["delta_low"] = (data.Low / data.Close) - 1

data["delta_high"] = (data.High / data.Close) - 1

data.drop(columns="Open", inplace=True)

data = data.iloc[30:].reset_index(drop=True)

data["vol_over_trend"] = np.where(data.Volume > data.volume_trend_30, 1, 0)

data["price_over_trend"] = np.where(data.Close > data.price_trend_60, 1, 0)

data["delta_vol_trend"] = (data.Volume / data.volume_trend_30) - 1

data["delta_price_trend"] = (data.Close / data.price_trend_60) - 1

print(data.shape)

dir = "./dataset"

if not os.path.exists(dir):
    os.mkdir(dir)

file = f"{dir}/fe_data_{ticker}.csv"

data.to_csv(file, index=False)
