"""
This script extract data from yfinance. 

TBD:
    - New feature: financial data by Q and Y -> add data to each date.
"""

from pandas_datareader import data as pdr

import pandas as pd


class Data_extract:
    """
    ETL process for stock price time serie data
    """

    def get_data(ticker: str, star_date, end_date) -> pd.DataFrame:
        """Get data from yfinance through pandas data reader lib."""

        ticker = ticker

        # download dataframe
        raw_data = pdr.get_data_yahoo(ticker, start=star_date, end=end_date)

        return raw_data
