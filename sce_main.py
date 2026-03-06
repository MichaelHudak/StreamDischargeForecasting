#!/usr/bin/env python
# coding: utf-8


# ## Import Statements


# In[2]:




import sys
import os


#get_ipython().system('{sys.executable} -m pip freeze > requirements.txt')


# In[41]:
print(sys.version)
print(sys.executable)


# In[31]:
from sce_functions import *
import pandas as pd
import numpy as np
import plotly.express as px


import matplotlib.pyplot as plt
import seaborn as sns
from dataretrieval import waterdata, nwis, utils
from datetime import date, datetime


# seasonality should be downloaded from GitHub link & in directory
from seasonality import seasonal_strength


import sktime
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.neuralforecast import NeuralForecastLSTM
from sktime.utils.plotting import plot_windows, plot_series
from sktime.forecasting.model_selection import ForecastingGridSearchCV
from sktime.performance_metrics.forecasting import MeanSquaredError


from sktime.split import (
    ExpandingWindowSplitter,
    SingleWindowSplitter,
    SlidingWindowSplitter,
)
from sktime.forecasting.model_evaluation import evaluate
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


from permetrics.regression import RegressionMetric


# LSTM libraries
from sktime.forecasting.neuralforecast import NeuralForecastLSTM
from sktime.split import temporal_train_test_split




# ## Define constants & codes


# In[ ]:


START_DATE= "2016-01-01"
END_DATE = "2025-01-01"
USGS_KEY = "SW1b2R5vFngjPzlWbq3XMQrboglYbpQQcdd1Wcc8"


# https://api.waterdata.usgs.gov/ogcapi/v0/openapi?f=html#/daily/SW1b2R5vFngjPzlWbq3XMQrboglYbpQQcdd1Wcc8


# In[ ]:




loc_stat_ids = {
    "USGS-392104077554801" : "31200", #gw site w/ readings at 12:00, ft
}




# In[1]:


data_source_dict = {
    'A' : {'stream': 'USGS-1571184', 'gw': 'USGS-400209077183301',
               'weather' : r"C:\Users\hudak\OneDrive - Washington College\SCE\SCE Weather Data\A_Biglerville_weather.csv"},
    'B' : {'stream': 'USGS-1567000', 'gw': 'USGS-402735077100901',
               'weather' : r"C:\Users\hudak\OneDrive - Washington College\SCE\SCE Weather Data\B_Lewistown_weather.csv"},
    'C' : {'stream': 'USGS-1550000', 'gw': 'USGS-412427076594401',
               'weather' : r"C:\Users\hudak\OneDrive - Washington College\SCE\SCE Weather Data\C_Williamsport_weather.csv"},
    'D' : {'stream': 'USGS-1526500', 'gw': 'USGS-420710077052101',
               'weather' : r"C:\Users\hudak\OneDrive - Washington College\SCE\SCE Weather Data\D_Corning_weather.csv"},
    'E' : {'stream': 'USGS-1514000', 'gw': 'USGS-420815076155501',
               'weather' : r"C:\Users\hudak\OneDrive - Washington College\SCE\SCE Weather Data\E_Binghamton_weather.csv"}
}




# In[ ]:


letter = "A"
stream_df = get_stream_data(letter["stream"])
gw_df = get_gw_data(letter["gw"])
combined_hydro_df = merge_dfs(gw_df, stream_df)
normalized_hydro_df = process_hydro_data(combined_hydro_df)


weather_df = process_weather_from_csv(letter["weather"])
combined_df = merge_hydro_weather(normalized_hydro_df, weather_df)


unsplit_df = include_gw(combined_df, yes_no=True)
unsplit_df_no_gw = include_gw(combined_df, yes_no=False)


y_train_val, y_test, X_train_val, X_test = data_split(unsplit_df, forecast_horizon=30)
fh_list = forecast_list(y_test)


cv = ExpandingWindowSplitter(initial_window =
                             int(len(y_train_val)-360),
                             fh=fh_list,
                             step_length = len(fh_list)
                             )
#cv = SingleWindowSplitter(fh=fh_list, window_length=len(y_train_val) - fh_list[-1])


## LSTM ONLY
gscv_lstm = set_lstm_test(cv)
gscv_lstm.fit(y_train_val, X=X_train_val, fh=fh_list)
print(f"Best parameters for {letter}: {gscv_lstm.best_params_}")


lstm_forecast_model = gscv_lstm.best_forecaster_
y_pred = lstm_forecast_model.predict()


lstm_scores = calc_all_metrics()
print(lstm_scores)




# In[ ]: ARIMAX ONLY
avg_X_all = avg_by_date(X_train_val)
future_X_values = find_future_X_values(y_test, avg_X_all)


gscv_arima = set_arima_gscv()
gscv_arima.fit(y_train_val, X=X_train_val, fh=fh_list)
y_arima_pred = gscv_arima.predict(X=future_X_values)


arima_scores = calc_all_metrics(y_test, y_arima_pred)
print(arima_scores)


print(gscv_arima.summary())


# In[ ]:






# In[ ]:




# In[ ]:




# fig_df = pd.merge(y_test, y_arima2_pred, how='inner', left_on='index', right_on=y_arima_pred.index)
# fig_df = fig_df.rename(columns={"log_discharge_x" : "Real Discharge",
#                        "log_discharge_y" : "Predicted Discharge"})
# print(fig_df.head())
# fig = px.scatter(fig_df, x=fig_df.index, y= ['Real Discharge', 'Predicted Discharge'])
# fig.show()




# ## Scaling
# Scaling can be done as part of model training, or I can pre-scale just the training data.


# In[ ]:
