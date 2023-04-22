""" This script will run if multistep was set on model.py """

import pandas as pd

class Multistep_reg:
    def lightgb_ms(
        self, X_train, X_test, y_train, regresor, period, predicted,
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

        y_pred = []

        for i in range(0,period):
    
            if i > 0:    
                for j in range(i,0,-1):
                    x_train.loc[len(x_train)-period+i-1,predicted] = y_pred[-j]
    
            y_pred.append(regresor.predict(X_test.iloc[i]))

        return y_pred
    
   
