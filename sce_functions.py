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
#from sktime.performance_metrics.forecasting import MeanSquaredError

#from sktime.forecasting.model_selection import ExpandingWindowSplitter, SingleWindowSplitter
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

def get_stream_data(site_id):
    global START_DATE, END_DATE

    df, metadata = waterdata.get_daily(
        monitoring_location_id = site_id,
        parameter_code = '00060', #discharge
        time = f"{START_DATE}/{END_DATE}"
    )

    condensed_df = df[['time', 'value']]
    return condensed_df


# In[ ]:


def get_gw_data(site_id):
    #df, metadata = nwis.get_gwlevels(sites=site_id, start=start_date, 
                                    #  end="2025-09-30"
                                    # )


    df, metadata = waterdata.get_daily(
        monitoring_location_id = site_id,
        parameter_code = 72019,
        time = f"{START_DATE}/{END_DATE}" # having a start date and end date is essential
    )
    #print(df.head())
    # select only time/value pairs that represent daily means (statistic_id=='00003')
    # daily maximimun (00001) and daily minimum (00002) are also options
    cleaned_df = df[['time', 'value']][df['statistic_id']==loc_stat_ids[site_id]] 
    return cleaned_df


# In[ ]:


def merge_dfs(gw_df, sw_df):
    gw = gw_df.rename(columns={'value':'gw_level'})
    sw = sw_df.rename(columns={'value':'discharge'})

    merged_df = pd.merge(gw, sw, on='time', how='right')

    return merged_df


# ## Test cleaning below \\I/
# gw_df = get_gw_data(gw_code)
# stream_df = get_stream_data(stream_code)
# combined_hydro_df = merge_dfs(gw_df, stream_df)
# ## Seasonality analysis

# In[ ]:


# downloaded from https://github.com/vcerqueira/blog/tree/main/src
#seasonal_strength(combined_df['discharge'], period=365)


# The authors of the seasonal_strength method recommend applying seasonal differencing if seasonal_strength is > 0.64.
# Because the seasonal_strength does not surpass the threshold and because seasonal variation will be accounted for with meteorological variables, no seasonal differencing will be applied.

# ## Preprocessing hydro data

# In[ ]:


# Pass in the combined_df that includes 'gw_level' and 'discharge'
def process_hydro_data(df, show_plots = False):
    gw = df['gw_level']
    dis = df['discharge']
    print(f"Skew value of groundwater: {gw.skew()},\n and discharge: {dis.skew()}")

    # Reflect gw to be right-skewed
    if gw.skew() < 0:
        reflected_gw = gw.max() - df['gw_level'] + 1

    log_gw = np.log(reflected_gw)
    log_dis = np.log(dis)
    print(f"Skew value of logarithmic groundwater: {log_gw.skew()},\n and logarithmic discharge: {log_dis.skew()}")

    if show_plots == True:
        print('We will display some plots')
        sns.histplot(gw, kde=True, bins=50)
        plt.title("Groundwater distribution")
        plt.show()

        sns.histplot(reflected_gw, kde=True, bins=50)
        plt.title("Reflected groundwater distribution")
        plt.show()

        sns.histplot(log_gw, kde=True, bins=50)
        plt.title("Logarithmic groundwater distribution")
        plt.show()

        sns.histplot(dis, kde=True, bins=50)
        plt.title("Discharge distribution")
        plt.show()

        sns.histplot(log_dis, kde=True, bins=50)
        plt.title("Logarithmic discharge distribution")
        plt.show()

    output_df = pd.concat([df['time'], log_gw, log_dis], axis=1)
    output_df = output_df.rename(columns = {'gw_level':'log_gw', 'discharge':'log_discharge'})
    output_df.sort_values(by='time')
    return output_df



# In[ ]:


#processed_hydro_df = process_hydro_data(combined_hydro_df, show_plots=False)


# In[ ]:


#print(processesd_hydro_df.isna().sum())


# At this point, log_dis and log_gw both underwent a logarthimic transform to normalize their skew from their most extreme values. gw_level had to be reflected first because logarithmic transformation shifts values to the left. Both skew coefficients are between -1 and 1, which suggests relative normality. Therefore, we can use MinMaxScaler() on these series.

# ## Read, process, and merge weather data

# In[ ]:


# Requires a csv path for each weather df
def process_weather_from_csv(csv_path):
    global START_DATE, END_DATE

    weather_df = pd.read_csv(rf"{csv_path}", header=1)

    #Converts to date object
    weather_df['Date'] = pd.to_datetime(weather_df['Date'])
    weather_df['Date'] = weather_df['Date'].dt.date

    weather_df = weather_df.loc[(weather_df['Date'] >= datetime.strptime(START_DATE, "%Y-%m-%d").date()) & 
                            (weather_df['Date'] <= datetime.strptime(END_DATE, "%Y-%m-%d").date())]

    return weather_df


# In[ ]:


# Merges and sorts dfs to enter model training
def merge_hydro_weather(hdf, wdf):
    unscaled_df = pd.merge(hdf, wdf, left_on='time', right_on='Date', how='left')
    unscaled_df = unscaled_df[['time', 'log_gw', 'log_discharge', 'TMAX (Degrees Fahrenheit)',
                                'PRCP (Inches)', 'SNOW (Inches)', 'SNWD (Inches)']]
    unscaled_df = unscaled_df.sort_values(by='time')
    unscaled_df.rename(columns={'time' : 'index'}, inplace=True)
    unscaled_df.set_index('index', inplace=True)
    unscaled_df.index = pd.to_datetime(unscaled_df.index)

    # Fills in nan values with linear average of nearby points
    # If the first or last value is NaN, ffill() and bfill() will cover 
    unscaled_df.interpolate(method='linear', inplace=True)
    unscaled_df = unscaled_df.ffill().bfill()

    return unscaled_df


# In[ ]:


def include_gw(df, yes_no):
    if yes_no == True:
        return df
    elif yes_no == False:
        return df.loc[:, df.columns != 'log_gw']


# In[ ]:


# csv_path = r''
# site_letter = ''
# weather_df = process_weather_from_csv(data_source_dict[site_letter]['weather'])
# unsplit_df = merge_hydro_weather(processed_hydro_df, weather_df)


# In[ ]:


# unsplit_df.head()


# In[ ]:


# na_counts = unsplit_df.isna().sum()
# print(na_counts)


# ## Data splitting
# Methods explained by: https://www.youtube.com/watch?v=27SGf2w62ic

# In[14]:


#def train

# Splits train and test data, kwwping the real test data out of model training
# Will still split X_train and y_train for validation
def data_split(unsplit_df):
    y_train_val, y_test, X_train_val, X_test = temporal_train_test_split(
                                y=unsplit_df['log_discharge'],
                                X=unsplit_df.loc[:, unsplit_df.columns != 'log_discharge'],
                                test_size = 30) #number of rows to include in test set
#X_train = X_train.drop(['dates'], axis=1)
# In[ ]:


def forecast_list(yt):
    forecast_length = 0
    fh_list = []
    for i in yt:
        forecast_length += 1
        fh_list.append(forecast_length)
    return fh_list


# In[ ]:


# Creates a validation set to score model, different from the test set
#cv = ExpandingWindowSplitter(initial_window = 365, fh=fh_list, step_length = 365) #about 1/3 of a year
# def set_splitter(fh_list, y_train):
#     cv = SingleWindowSplitter(fh=fh_list, window_length=len(y_train) - fh_list[-1])
#     plot_windows(cv=cv, y=y_train)
#     cv.get_n_splits(y=y_train)
#     return cv
# ## LSTM

# In[ ]:


# https://stackoverflow.com/questions/63903016/calculate-nash-sutcliff-efficiency
def nse(predictions, targets):
    return 1 - (np.sum((targets - predictions) ** 2) / np.sum((targets - np.mean(targets)) ** 2))


# In[ ]:


# https://permetrics.readthedocs.io/en/v2.0.0/pages/regression/NSE.html
def permetric_nse(y_true, y_pred):
    #y_true = np.array(y_true)
    #y_pred = np.array(y_pred)

    evaluator = RegressionMetric(y_true, y_pred)
    print(evaluator.nash_sutcliffe_efficiency())

    evaluator = RegressionMetric(y_true, y_pred)
    print(evaluator.NSE(multi_output="raw_values"))


# In[ ]:


# https://permetrics.readthedocs.io/en/v2.0.0/pages/regression/KGE.html
def permetric_kge(y_true, y_pred):
    #y_true = np.array(y_true)
    #y_pred = np.array(y_pred)

    evaluator = RegressionMetric(y_true, y_pred)
    print(evaluator.kling_gupta_efficiency())

    evaluator = RegressionMetric(y_true, y_pred)
    print(evaluator.KGE(multi_output="raw_values"))

# In[ ]:

# Calculates all listed regression metrics 
def calc_all_metrics(y_true, y_pred):
    evaluator = RegressionMetric(y_true, y_pred)

    results = evaluator.calculate_all_metrics(metrics=["MAE", "RMSE", "R2", "MAPE", "KGE", "NSE"])
    return results

def set_lstm_test():
    lstm = NeuralForecastLSTM(  
        local_scaler_type = 'minmax',
        futr_exog_list= None, # we would not know future values, so the model shouldn't
        verbose_fit = True,
        verbose_predict = True,
        #early_stop_patience_steps = 1,
        #val_check_steps = 1,
        input_size = -1, # uses all historical data
        #encoder_n_layers = 2,
        #encoder_hidden_size = 200,
        max_steps = 100
    )
    param_grid = {
        'encoder_n_layers' : [2,3],
        'encoder_hidden_size' : [200, 300, 400],
        'batch_size' : [64, 128],
    }

    gscv = ForecastingGridSearchCV(
        forecaster=lstm,
        param_grid=param_grid,
        cv=cv,
        verbose=2,
        scoring=MeanSquaredError(square_root=True), # RMSE
        error_score='raise',
    )

    return gscv
# In[ ]:


def gscv_lstm():
    lstm = NeuralForecastLSTM(  
        local_scaler_type = 'minmax',
        futr_exog_list= None, # we would not know future values, so the model shouldn't
        verbose_fit = True,
        verbose_predict = True,
        #early_stop_patience_steps = 1,
        #val_check_steps = 1,
        input_size = -1, # uses all historical data
        #encoder_n_layers = 2,
        #encoder_hidden_size = 200,
        max_steps = 100
    )

    param_grid = {
        'encoder_n_layers' : [2,3],
        'encoder_hidden_size' : [200, 300, 400],
        'batch_size' : [64, 128],
    }

    gscv = ForecastingGridSearchCV(
        forecaster=lstm,
        param_grid=param_grid,
        cv=cv,
        verbose=2,
        scoring=MeanSquaredError(square_root=True), # RMSE
        error_score='raise',
    )

    return gscv
