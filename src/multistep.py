""" This script will run if multistep was set on model.py """

import os

import numpy as np

import pandas as pd

from feature_engineering import Feature_engineering as fe

from sklearn import metrics


class Multistep_reg:
    def lightgb_ms(
        self, X_train, X_test, y_train, y_test, regresor, period, params
    ) -> pd.DataFrame:
        """
        _summary_

        Args:
            X_train (_type_): _description_
            X_test (_type_): _description_
            y_train (_type_): _description_
            y_test (_type_): _description_
            regresor (_type_): _description_
            period (_type_): _description_
            params (_type_): _description_

        Returns:
            pd.DataFrame: _description_
        """
        regresor.fit(X_train, y_train)

        for i in np.arange(period):

            y_pred = np.array(period)

            y_pred[i] = regresor.predict(X_test[i])

        return metrics.mean_absolute_error(y_pred, y_test)
