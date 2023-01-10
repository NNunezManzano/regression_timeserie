""" This script will run if multistep was set on model.py """

import os

import numpy as np

import pandas as pd

from feature_engineering import Feature_engineering as fe


class Multistep_reg:
    def lightgb_ms(
        self, X_train, X_test, y_train, y_test, regresor, period
    ) -> pd.DataFrame:

        for i in np.arange(period):

            regresor.fit(X_train, y_train)

            y_pred = regresor.predict(X_test)
