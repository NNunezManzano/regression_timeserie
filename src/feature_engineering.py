"""
    Feature engineering over int variables of a time serie dataset.
    
    lags
    
    trends
    
    stocks_ts
"""
import pandas as pd

import numpy as np


class Feature_engineering:
    def lags(self, dataframe: pd.DataFrame, n_lags: int, column: str) -> pd.DataFrame:
        """
        Add "n" column with the close price of the previous "n" rows.

        Args:
            dataframe (pd.DataFrame): it must include the column "Close" or "Price"
            n_lags (int): numbers of lags
            column (str): column to be laged

        Returns:
            pd.DataFrame: input dataframe with n_lags and delta_lags extra columns.
        """
        for n in np.arange(1, n_lags + 1):
            col_name = f"{column}_lag_{n}"  # set new column name
            dataframe[col_name] = dataframe[column].shift(
                n
            )  # get "n" previous row value

        dataframe.fillna(0, inplace=True)

        # calculate delta between curr value and lag n
        for n in np.arange(1, n_lags + 1):
            lag_name = f"{column}_lag_{n}"
            col_name = f"delta_{column}_{n}"
            dataframe[col_name] = (dataframe[column] - dataframe[lag_name]) / dataframe[
                column
            ]  # percentage variation from lag

        return dataframe

    def trend(self, dataframe: pd.DataFrame, trend: int, column: str) -> pd.DataFrame:
        """
        Generate a trend of n days for the CLOSE PRICE of each row.

        Args:
            dataframe (pd.DataFrame): must include column "Close"
            trend (int): number of days backward to create the trend
            column (str): column to calculate trend of

        Returns:
            pd.DataFrame: dataframe with an extra column named "{column}_trend_{trend}"
        """
        colname = f"{column}_trend_{trend}"

        try:
            if len(dataframe[colname]) > 0:
                raise NameError(f"Column '{colname}' already exist")
        except:
            pass

        # take n periods in the past and get the mean value
        dataframe[colname] = (
            dataframe[column]
            .rolling(
                window=trend,
                min_periods=1,
                center=False,  # center must be set as "False" to not get data of the future.
            )
            .mean()
        )

        # Now set if the current value is over or under the trend and calculate delta
        overcol = f"{column[:3]}_over_trend"

        deltacol = f"{column[:3]}_delta_trend"

        # As it is only using the "lemma" of the column name, we have to check if it is already exist
        try:
            if len(dataframe[overcol]) > 0:
                overcol = f"{column[:4]}_over_trend"
        except:
            pass

        try:
            if len(dataframe[deltacol]) > 0:
                deltacol = f"{column[:4]}_delta_trend"
        except:
            pass

        try:
            if len(dataframe[overcol]) > 0:
                raise NameError(f"Column '{overcol}' already exist")
        except:
            pass

        try:
            if len(dataframe[deltacol]) > 0:
                raise NameError(f"Column '{deltacol}' already exist")
        except:
            pass

        dataframe[overcol] = np.where(dataframe[column] > dataframe[colname], 1, 0)

        dataframe[deltacol] = (dataframe[column] / dataframe[colname]) - 1

        return dataframe

    def stocks_ts(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Creates new fueatures base on a OHLC time serie dataset.

        Args:
            dataframe (pd.DataFrame): must include column ,"Open", "High", "Low", "Close", "Volume".
        """
        dataframe["daily_market_cap"] = dataframe.Close * dataframe.Volume
        dataframe["delta_low"] = (dataframe.Low / dataframe.Close) - 1
        dataframe["delta_high"] = (dataframe.High / dataframe.Close) - 1
        dataframe.drop(columns="Open", inplace=True)
