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
#from seasonality import seasonal_strength

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
from sktime.split import temporal_train_test_split

# ## Define constants & codes
START_DATE= "2016-01-01"
END_DATE = "2025-01-01"
USGS_KEY = "SW1b2R5vFngjPzlWbq3XMQrboglYbpQQcdd1Wcc8"

# https://api.waterdata.usgs.gov/ogcapi/v0/openapi?f=html#/daily/SW1b2R5vFngjPzlWbq3XMQrboglYbpQQcdd1Wcc8


# In[ ]:
loc_stat_ids = {
    #gw site w/ readings at 12:00, ft -- 31200
    # Depth to water level, feet below land surface -- 72019
    'USGS-400209077183301' : "00003", 
    'USGS-402735077100901' : "00003", 
    'USGS-412427076594401' : "00003", 
    'USGS-420710077052101' : "00003", 
    'USGS-420815076155501' : "00003", 
}

data_source_dict = {
    'A' : {'stream': 'USGS-01571184', 'gw': 'USGS-400209077183301',
               'weather' : os.path.join('weather_data', 'A_Biglerville_weather.csv')},
    'B' : {'stream': 'USGS-01567000', 'gw': 'USGS-402735077100901',
               'weather' : os.path.join('weather_data', 'B_Lewistown_weather.csv')},
    'C' : {'stream': 'USGS-01550000', 'gw': 'USGS-412427076594401',
               'weather' : os.path.join('weather_data', 'C_Williamsport_weather.csv')},
    'D' : {'stream': 'USGS-01526500', 'gw': 'USGS-420710077052101',
               'weather' : os.path.join('weather_data', 'D_Corning_weather.csv')},
    'E' : {'stream': 'USGS-01514000', 'gw': 'USGS-420815076155501',
               'weather' : os.path.join('weather_data', 'E_Binghamton_weather.csv')}
}

# In[ ]:

letter = "A"
stream_df = get_stream_data(data_source_dict[letter]["stream"])
print(stream_df.head())

gw_df = get_gw_data(data_source_dict[letter]["gw"])
print(gw_df.head())

combined_hydro_df = merge_dfs(gw_df, stream_df)
normalized_hydro_df = process_hydro_data(combined_hydro_df)
print(normalized_hydro_df.head())

weather_df = process_weather_from_csv(data_source_dict[letter]["weather"])
combined_df = merge_hydro_weather(normalized_hydro_df, weather_df)
print(combined_df.head())

unsplit_df = include_gw(combined_df, yes_no=True)
unsplit_df_no_gw = include_gw(combined_df, yes_no=False)

### GROUNDWATER INCLUDED MODELS \/
y_train_val_gw, y_test_gw, X_train_val_gw, X_test_gw = data_split(unsplit_df, forecast_horizon=30)
fh_list_gw = forecast_list(y_test_gw)

cv_gw = ExpandingWindowSplitter(initial_window =
                             int(len(y_train_val_gw)-360),
                             fh=fh_list_gw,
                             step_length = len(fh_list_gw)
                             )
#cv = SingleWindowSplitter(fh=fh_list, window_length=len(y_train_val_gw) - fh_list[-1])
avg_X_all_gw = avg_by_date(X_train_val_gw)
future_X_values_gw = find_future_X_values(y_test_gw, avg_X_all_gw)

## LSTM ONLY
gscv_lstm_gw = set_lstm_test(cv_gw)
print("Fitting LSTM model...")
gscv_lstm_gw.fit(y_train_val_gw, X=X_train_val_gw, fh=fh_list_gw)
print(f"\n\n\nBest parameters for groundwater LSTM {letter}: {gscv_lstm_gw.best_params_}")

gw_lstm_forecast_model = gscv_lstm_gw.best_forecaster_
y_lstm_pred_gw = gw_lstm_forecast_model.predict(X=future_X_values_gw)

lstm_scores_gw = calc_all_metrics(y_test_gw, y_lstm_pred_gw)
print("\n\nLSTM Scores including groundwater:")
print(lstm_scores_gw)

## ARIMAX ONLY

arima_gw = set_arima()
print("\nValidating groundwater ARIMA model...")
X_train_val_gw_num = X_train_val_gw.select_dtypes(include=['number'])
results_arima_gw = evaluate_arima(arima_gw, y_train_val_gw, cv_gw, X=X_train_val_gw_num)

print("\nValidation RMSE per split:")
print(results_arima_gw["test_MeanSquaredError"])
print(f"\nMean RMSE: {results_arima_gw['test_MeanSquaredError'].mean():.4f}")

print("\nFitting groundwater ARIMA model...")
arima_gw.fit(y_train_val_gw, X=X_train_val_gw_num, fh=fh_list_gw)
y_arima_pred_gw = arima_gw.predict(X=future_X_values_gw)

arima_scores_gw = calc_all_metrics(y_test_gw, y_arima_pred_gw)
print("\n\nGroundwater ARIMA Scores:")
print(arima_scores_gw)
print("\n\nGroundwater ARIMA Model Summary:")
print(arima_gw.summary())



### GROUNDWATER ABSENT MODELS \/
y_train_val_no, y_test_no, X_train_val_no, X_test_no = data_split(unsplit_df_no_gw, forecast_horizon=30)
fh_list_no = forecast_list(y_test_no)

cv_no = ExpandingWindowSplitter(initial_window =
                             int(len(y_train_val_no)-360),
                             fh=fh_list_no,
                             step_length = len(fh_list_no)
                             )
#cv = SingleWindowSplitter(fh=fh_list, window_length=len(y_train_val_gw) - fh_list[-1])
avg_X_all_no = avg_by_date(X_train_val_no)
future_X_values_no = find_future_X_values(y_test_no, avg_X_all_no)


## LSTM ONLY
gscv_lstm_no = set_lstm_test(cv_no)
print("Fitting LSTM model...")
gscv_lstm_no.fit(y_train_val_no, X=X_train_val_no, fh=fh_list_no)
print(f"Best parameters for non-groundwater LSTM {letter}: {gscv_lstm_no.best_params_}")

gw_lstm_forecast_model_no = gscv_lstm_no.best_forecaster_
y_lstm_pred_no = gw_lstm_forecast_model_no.predict(X=future_X_values_no)

lstm_scores_no = calc_all_metrics(y_test_no, y_lstm_pred_no)
print("LSTM Scores excluding groundwater:")
print(lstm_scores_no)

# In[ ]: ARIMAX ONLY
arima_no = set_arima()
print("\nValidating non-groundwater ARIMA model...")
X_train_val_no_num = X_train_val_no.select_dtypes(include=['number'])
results_arima_no = evaluate_arima(arima_no, y_train_val_no, cv_no, X=X_train_val_no_num)

print("\nValidation RMSE per split:")
print(results_arima_no["test_MeanSquaredError"])
print(f"\nMean RMSE: {results_arima_no['test_MeanSquaredError'].mean():.4f}")


print("Fitting non-groundwater ARIMA model...")
arima_no.fit(y_train_val_no, X=X_train_val_no_num, fh=fh_list_no)
y_arima_pred_no = arima_no.predict(X=future_X_values_no)

arima_scores_no = calc_all_metrics(y_test_no, y_arima_pred_no)
print("Non-Groundwater ARIMA Scores:")
print(arima_scores_no)
print("\nARIMA Model Summary:")
print(arima_no.summary())



moving_average_plot(combined_df, window_size=30)
forecast_vs_actual_plot(y_test_gw, y_lstm_pred_gw, y_arima_pred_gw, gw_included=True)
forecast_vs_actual_plot(y_test_no, y_lstm_pred_no, y_arima_pred_no, gw_included=False)
compare_forecasts_plots(y_test_gw, y_lstm_pred_gw, y_lstm_pred_no, model_type="LSTM")
compare_forecasts_plots(y_test_gw, y_arima_pred_gw, y_arima_pred_no, model_type="ARIMA")

save_run_results(letter, results_arima_gw, results_arima_no, 
                arima_gw, arima_no, 
                lstm_scores_gw, lstm_scores_no,
                arima_scores_gw, arima_scores_no)

# save_data(letter, pre_model_df=combined_df, y_true=y_test_gw, 
#           y_lstm_pred_gw=y_lstm_pred_gw, y_arima_pred_gw=y_arima_pred_gw, 
#           y_lstm_pred_no=y_lstm_pred_no, y_arima_pred_no=y_arima_pred_no, X_true=X_test_gw)






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
