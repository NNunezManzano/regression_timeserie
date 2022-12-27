"""
ETL.py

Input: Get data from yfinance throw pandas_datareader lib.

Output: .csv file.

This script handle the ETL process, gets data from yfinance, adds lags and trends, and loads to a .csv file. 
It's could be run in docker or GCP, changing the working directory.    
"""

from pandas_datareader import data as pdr

import os

import numpy as np

import pandas as pd

import yfinance as yf

# set working directory
os.chdir("/home/nnm/Documents/Proyectos/repos/regression_timeserie")

yf.pdr_override()

ticker = "SPY"

# download dataframe
raw_data = pdr.get_data_yahoo(ticker, start="2007-01-01", end="2017-12-31")


def lags_price(dataframe: pd.DataFrame, n_lags: int) -> pd.DataFrame:
    """
    For each row it add "n" column with the close price of the previous "n" rows.

    Args:
        dataframe (pd.DataFrame): it must include the column "Close" or "Price"
        n_lags (int): numbers of lags

    Returns:
        pd.DataFrame: input dataframe with n_lags and delta_lags extra columns.
    """
    for n in np.arange(1, n_lags + 1):
        col_name = f"close_lag_{n}"
        dataframe[col_name] = dataframe.Close.shift(n)  # get "n" previous row value

    dataframe.fillna(0, inplace=True)

    for n in np.arange(1, n_lags + 1):
        lag_name = f"close_lag_{n}"
        col_name = f"delta_close_{n}"
        dataframe[col_name] = (
            dataframe.Close - dataframe[lag_name]
        ) / dataframe.Close  # percentage variation from lag

    return dataframe


def lags_volume(dataframe: pd.DataFrame, n_lags: int) -> pd.DataFrame:
    """
    For each row it add "n" column with the volume of the previous "n" rows.

    Args:
        dataframe (pd.DataFrame): it must include the column "Volume"
        n_lags (int): numbers of lags

    Returns:
        pd.DataFrame: input dataframe with n_lags and delta_lags extra columns.
    """
    for n in np.arange(1, n_lags + 1):
        col_name = f"volume_lag_{n}"
        dataframe[col_name] = dataframe.Volume.shift(n)  # get "n" previous row value

    dataframe.fillna(0, inplace=True)

    for n in np.arange(1, n_lags + 1):
        lag_name = f"volume_lag_{n}"
        col_name = f"delta_volume_{n}"
        dataframe[col_name] = (
            dataframe.Volume - dataframe[lag_name]
        ) / dataframe.Volume  # percentage variation from lag

    return dataframe


def trend_price(dataframe: pd.DataFrame, trend: int) -> pd.DataFrame:
    """
    Generate a trend of n days for the CLOSE PRICE of each row.

    Args:
        dataframe (pd.DataFrame): must include column "Close"
        trend (int): number of days backward to create the trend

    Returns:
        pd.DataFrame: dataframe with an extra column named "price_trend_{trend}"
    """
    colname = f"price_trend_{trend}"
    # take n periods in the past and get the mean value
    dataframe[colname] = dataframe.Close.rolling(
        window=trend,
        min_periods=1,
        center=False,  # center must be set as "False" to not get data of the future.
    ).mean()

    return dataframe


def trend_volume(dataframe: pd.DataFrame, trend: int) -> pd.DataFrame:
    """
    Generate a trend of n days for the VOLUME of each row.

    Args:
        dataframe (pd.DataFrame): must include column "Volume"
        trend (int): number of days backward to create the trend

    Returns:
        pd.DataFrame: dataframe with an extra column named "volume_trend_{trend}"
    """
    colname = f"volume_trend_{trend}"

    # take n periods in the past and get the mean value
    dataframe[colname] = dataframe.Volume.rolling(
        window=trend,
        min_periods=1,
        center=False,  # must be set as "False" to not get data of the future.
    ).mean()

    return dataframe


def data_transform(
    dataframe: pd.DataFrame,
    price_lag: int = 0,
    price_trend: int = 0,
    volume_lag: int = 0,
    volum_trend: int = 0,
) -> pd.DataFrame:
    """
    Transform raw data to a new data frame with trends and lags.

    Args:
        dataframe (pd.DataFrame):    Must inlcude Close and Volume rows.
        price_lag (int, optional):   Number of lags to apply on price column.   (Defaults to 0)
        price_trend (int, optional): Days backward to fill the price trend.     (Defaults to 0)
        volume_lag (int, optional):  Number of lags to apply on volume column.  (Defaults to 0)
        volum_trend (int, optional): Days backward to fill the volume trend.    (Defaults to 0)

    Returns:
        pd.DataFrame: dataframe with lags and trends of price and volume.
    """

    if price_lag != 0:
        dataframe = lags_price(dataframe, price_lag)

    if price_trend != 0:
        dataframe = trend_price(dataframe, price_trend)

    if volume_lag != 0:
        dataframe = lags_volume(dataframe, volume_lag)

    if volum_trend != 0:
        dataframe = trend_volume(dataframe, volum_trend)

    return dataframe


data = data_transform(raw_data, 30, 60, 15, 30)


dir = "./dataset"

if not os.path.exists(dir):  # create the directory if it doesn't exist
    os.mkdir(dir)

file = f"{dir}/data_{ticker}.csv"  # file name

data.to_csv(file)
