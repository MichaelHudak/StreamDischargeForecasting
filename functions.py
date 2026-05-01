import os
import pandas as pd
import numpy as np
import plotly.express as px
import json
from pathlib import Path

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

# ## Define constants & codes
START_DATE= "2010-01-01"
END_DATE = "2025-01-01"
USGS_KEY = "SW1b2R5vFngjPzlWbq3XMQrboglYbpQQcdd1Wcc8"

loc_stat_ids = {
    #gw site w/ readings at 12:00, ft -- 31200
    # Depth to water level, feet below land surface -- 72019
    'USGS-400209077183301' : "00003", 
    'USGS-402735077100901' : "00003", 
    'USGS-412427076594401' : "00003", 
    'USGS-420710077052101' : "00003", 
    'USGS-420815076155501' : "00003", 
}

def get_stream_data(site_id):
    global START_DATE, END_DATE

    df, metadata = waterdata.get_daily(
        monitoring_location_id = site_id,
        parameter_code = '00060', #discharge
        time = f"{START_DATE}/{END_DATE}"
    )


    condensed_df = df[['time', 'value']]
    return condensed_df


def get_gw_data(site_id):
    #df, metadata = nwis.get_gwlevels(sites=site_id, start=start_date,
                                    #  end="2025-09-30"
                                    # )

    df, metadata = waterdata.get_daily(
        monitoring_location_id = site_id,
        parameter_code = '72019', # depth to water level, feet below land surface
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

# Pass in the combined_df that includes 'gw_level' and 'discharge'
def process_hydro_data(df, show_plots = False):
    gw = df['gw_level']
    dis = df['discharge']
    print(f"Skew value of groundwater: {gw.skew()},\n and discharge: {dis.skew()}")

    gw_log_transform = False
    dis_log_transform = False

    # Reflect gw to be right-skewed
    if gw.skew() < -.75: # if the skew is less than -0.75, we will reflect it to be right-skewed, 
        reflected_gw = gw.max() - df['gw_level'] + 1
        gw_col = np.log(reflected_gw)
        gw_log_transform = True
    elif gw.skew() > .75: # if the skew is greater than 0.75, we will log transform it to be less skewed, but it is already right-skewed so we won't reflect it
        gw_col = np.log(gw)
        gw_log_transform = True
    else:
        print("Groundwater skew is too low for transformation.")
        gw_col = gw

    if dis.skew() < -.75: # if the skew is less than -0.75, we will reflect it to be right-skewed
        reflected_dis = dis.max() - df['discharge'] + 1
        dis_col = np.log(reflected_dis)
        dis_log_transform = True
    elif dis.skew() > .75: # if the skew is greater than 0.75, we will log transform it to be less skewed
        dis_col = np.log(dis)
        dis_log_transform = True
    else:
        print("Discharge skew is too low for transformation.")
        dis_col = dis
    
    if gw_log_transform == True:
        print(f"Skew value of logarithmic groundwater: {gw_col.skew()}")
              
    if dis_log_transform == True:
        print(f"Skew value of logarithmic discharge: {dis_col.skew()}")


    if show_plots == True: # There may be a bug here based on the reflecting and log transforming
        
        sns.histplot(gw, kde=True, bins=50)
        plt.title("Groundwater distribution")
        plt.show()

        if gw_log_transform == True:
            sns.histplot(reflected_gw, kde=True, bins=50)
            plt.title("Reflected groundwater distribution")
            plt.show()

            sns.histplot(gw_col, kde=True, bins=50)
            plt.title("Logarithmic groundwater distribution")
            plt.show()

        if dis_log_transform == True:
            sns.histplot(dis, kde=True, bins=50)
            plt.title("Discharge distribution")
            plt.show()

            sns.histplot(dis_col, kde=True, bins=50)
            plt.title("Logarithmic discharge distribution")
            plt.show()


    output_df = pd.concat([df['time'], gw_col, dis_col], axis=1)
    # if gw_log_transform == True:
    #     output_df = output_df.rename(columns = {'gw_level':'log_gw'})
    # if dis_log_transform == True:
    #     output_df = output_df.rename(columns = {'discharge':'log_discharge'})
    output_df['time'] = pd.to_datetime(output_df['time'])
    output_df['time'] = output_df['time'].dt.date
    output_df.sort_values(by='time', inplace=True)
    return output_df


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

# downloaded from https://github.com/vcerqueira/blog/tree/main/src
#print(f"Seasonal strength of {letter}: {seasonal_strength(combined_hydro_df['discharge'], period=365)}")


# Merges and sorts dfs to enter model training
def merge_hydro_weather(hdf, wdf):
    unscaled_df = pd.merge(hdf, wdf, left_on='time', right_on='Date', how='left')
    unscaled_df = unscaled_df[['time', 'gw_level', 'discharge', 'TMAX (Degrees Fahrenheit)',
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


def include_gw(df, yes_no):
    if yes_no == True:
        return df
    elif yes_no == False:
        return df.loc[:, df.columns != 'gw_level']



# Splits train and test data, kwwping the real test data out of model training
# Will still split X_train and y_train for validation
def data_split(unsplit_df, forecast_horizon):
    y_train_val, y_test, X_train_val, X_test = temporal_train_test_split(
                                y=unsplit_df['discharge'],
                                X=unsplit_df.loc[:, unsplit_df.columns != 'discharge'],
                                test_size = forecast_horizon) #number of rows to include in test set
    return y_train_val, y_test, X_train_val, X_test


def forecast_list(yt): # Returns a list starting from 1 and counting up by 1
    return list(range(1, len(yt) + 1))


# https://stackoverflow.com/questions/63903016/calculate-nash-sutcliff-efficiency
def nse(predictions, targets):
    return 1 - (np.sum((targets - predictions) ** 2) / np.sum((targets - np.mean(targets)) ** 2))


# https://permetrics.readthedocs.io/en/v2.0.0/pages/regression/NSE.html
def permetric_nse(y_true, y_pred):
    #y_true = np.array(y_true)
    #y_pred = np.array(y_pred)


    evaluator = RegressionMetric(y_true, y_pred)
    print(evaluator.nash_sutcliffe_efficiency())


    evaluator = RegressionMetric(y_true, y_pred)
    print(evaluator.NSE(multi_output="raw_values"))


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
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    evaluator = RegressionMetric(y_true, y_pred)
    dict_result = evaluator.get_metrics_by_list_names(["MAE", "RMSE", "R2", "MAPE", "KGE", "NSE"])
    
    return dict_result


def set_lstm_test(cv, gw_included):
    if gw_included == True:
        futr_exog_list= ['gw_level', 'TMAX (Degrees Fahrenheit)',
                                'PRCP (Inches)', 'SNOW (Inches)', 'SNWD (Inches)']
    else:
        futr_exog_list= ['TMAX (Degrees Fahrenheit)',
                                'PRCP (Inches)', 'SNOW (Inches)', 'SNWD (Inches)']
    
    lstm = NeuralForecastLSTM(  
        local_scaler_type = 'minmax',
        futr_exog_list = futr_exog_list,
        verbose_fit = True,
        verbose_predict = True,
        input_size = 365, # uses past year of data
        batch_size=32,
        #encoder_n_layers = 2,
        #encoder_hidden_size = 200,
        learning_rate=0.005,
        max_steps = 200, # Relatively low to keep runtime manageable
    )

    param_grid = {
        'encoder_n_layers' : [2],
        'encoder_hidden_size' : [128],
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


# Returns a dataframe of daily averages for each input variable
def avg_by_date(training_df):
    training_df_copy = training_df.copy()
    training_df_copy = training_df_copy.drop(columns=['discharge'], errors='ignore') # removes discharge column, doesn't do anything if discharge is not present
    training_df_copy['dates'] = pd.to_datetime(training_df_copy.index)
    daily_avg = training_df_copy.groupby(
        [training_df_copy['dates'].dt.month, training_df_copy['dates'].dt.day]).mean()
    daily_avg.index.names = ['Month', 'Day']
    daily_avg = daily_avg.drop('dates', axis=1)
    return daily_avg


def find_future_X_values(y_test, avg_X_all):
    # Build list of (month, day) tuples for forecast horizon
    md_tuples = [(d.month, d.day) for d in y_test.index]

    # Select matching rows in one shot
    avg_X_forecast = avg_X_all.loc[md_tuples].copy()

    # Assign forecast dates as index
    avg_X_forecast.index = y_test.index
    return avg_X_forecast


def set_arima():
    arima = AutoARIMA(
        sp=365,
        start_P=0,  # no seasonal AR terms
        start_Q=0,    # no seasonal MA terms
        max_P=0,    # no seasonal AR terms
        max_Q=0,    # no seasonal MA terms
        max_D=1,    # allow seasonal differencing only
        max_p=2,    # looks back at most 2 most recent values
        max_q=2,    # looks back at most 2 most recent errors, 
        max_d=1,    # allows differencing to make the series stationary
        stepwise=True,
        information_criterion='aicc',
    )
    return arima

def evaluate_arima(arima, y_train_val, cv, X=None):
    results = evaluate(
        forecaster=arima,
        y=y_train_val,
        cv=cv,
        X=X,
        scoring=MeanSquaredError(square_root=True),
        return_data=True,
    )
    return results

# Moving average plot
def moving_average_plot(letter, df, window_size):
    os.makedirs(f'results/{letter}/plots', exist_ok=True)
    df['moving_average'] = df['discharge'].rolling(window=window_size).mean()
    sns.lineplot(x=df.index, y=df['discharge'], label='Discharge')
    sns.lineplot(x=df.index, y=df['moving_average'], label=f'Moving Average (window={window_size})')
    plt.title(f'Discharge and Moving Average {letter}')
    plt.xlabel('Date')
    plt.ylabel('Discharge')
    plt.legend()
    plt.savefig(f'results/{letter}/plots/moving_average_window_{window_size}_{letter}.png', dpi=300, bbox_inches='tight')
    plt.show()

# Forecast vs actual plot
def forecast_vs_actual_plot(letter, y_true, y_lstm_pred, y_arima_pred, gw_included=False):
    if gw_included:
        title_suffix = "with Groundwater"
    else:
        title_suffix = "without Groundwater"
    
    os.makedirs(f'results/{letter}/plots', exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=y_true.index, y=y_true.values, label='Actual Discharge')
    sns.lineplot(x=y_lstm_pred.index, y=y_lstm_pred.values, label='LSTM Predicted Discharge')
    sns.lineplot(x=y_arima_pred.index, y=y_arima_pred.values, label='ARIMAX Predicted Discharge')
    plt.title(f'Forecast {title_suffix} vs Actual Discharge {letter}')
    plt.xlabel('Date')
    plt.ylabel('Discharge')
    plt.legend()
    plt.savefig(f'results/{letter}/plots/forecast_vs_actual_{title_suffix.lower().replace(" ", "_")}_{letter}.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_forecasts_plots(letter, y_true, gw_included, gw_absent, model_type):
    os.makedirs(f'results/{letter}/plots', exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=y_true.index, y=y_true.values, label='Actual Discharge')
    sns.lineplot(x=gw_included.index, y=gw_included.values, label=f'{model_type} with Groundwater')
    sns.lineplot(x=gw_absent.index, y=gw_absent.values, label=f'{model_type} without Groundwater')
    plt.title(f'Comparison of {model_type} Forecasts with and without Groundwater {letter}')
    plt.xlabel('Date')
    plt.ylabel('Discharge')
    plt.legend()
    plt.savefig(f'results/{letter}/plots/compare_forecasts_{model_type.lower().replace(" ", "_")}_{letter}.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_data(letter, pre_model_df, y_true, y_lstm_pred_gw, y_arima_pred_gw, y_lstm_pred_no, y_arima_pred_no, X_true):
    os.makedirs(f"results/{letter}", exist_ok=True)
    pre_model_df.to_csv(f"results/{letter}/{letter}_pre_model_data.csv")
    forecast_df = pd.DataFrame({'y_true': y_true, 'y_lstm_pred_gw': y_lstm_pred_gw, 'y_arima_pred_gw': y_arima_pred_gw, 
                                'y_lstm_pred_no': y_lstm_pred_no, 'y_arima_pred_no': y_arima_pred_no})
    forecast_df = pd.concat([forecast_df, X_true], axis=1)
    forecast_df.to_csv(f"results/{letter}/{letter}_forecast_data.csv")


def save_run_results(letter, results_arima_gw, results_arima_no, 
                     arima_gw, arima_no, 
                     lstm_scores_gw, lstm_scores_no,
                     arima_scores_gw, arima_scores_no):
    
    out_dir = Path(f"results/{letter}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validation CV results
    results_arima_gw.to_csv(out_dir / "arima_gw_cv_results.csv")
    results_arima_no.to_csv(out_dir / "arima_no_cv_results.csv")

    # Model summaries
    with open(out_dir / "arima_gw_summary.txt", "w") as f:
        f.write(str(arima_gw.summary()))
    with open(out_dir / "arima_no_summary.txt", "w") as f:
        f.write(str(arima_no.summary()))

    # All scores in one place
    scores = {
        "lstm_gw": lstm_scores_gw,
        "lstm_no": lstm_scores_no,
        "arima_gw": arima_scores_gw,
        "arima_no": arima_scores_no,
    }
    with open(out_dir / "all_scores.json", "w") as f:
        json.dump(scores, f, indent=2)
