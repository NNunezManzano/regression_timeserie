"""
Input: Get data from yfinance throw pandas_datareader.

Output: .csv file.

This script handle the ETL process, gets data from yfinance and loads to a .csv file. 

TBD:
    - New feature: financial data by Q and Y -> add data to each date.
"""

from pandas_datareader import data as pdr

import os

import numpy as np

import pandas as pd

import yfinance as yf

# set working directory
try:
    os.chdir("../regression_timeserie")
except:
    os.chdir("../")

yf.pdr_override()

ticker = "SPY"

# download dataframe
raw_data = pdr.get_data_yahoo(ticker, start="2007-01-01", end="2017-12-31")

dir = "./dataset"

if not os.path.exists(dir):  # create the directory if it doesn't exist
    os.mkdir(dir)

file = f"{dir}/data_{ticker}.csv"  # file name

data.to_csv(file)
