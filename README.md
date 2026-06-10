# Stream Discharge Forecasting in the Susquehanna River Basin

### Summary
As part of my Senior Capstone requirements at Washington College, I developed machine learning models to forecast stream discharge. The forecasting models used include Long Short Term Memory (LSTM) and AutoRegressive Integrated Moving Average (ARIMA). I developed two variations of each model for each of three sites in the Susquehanna River Basin (SRB); that is, each location had two LSTM models and two ARIMA models. The two versions of each model were based on the inclusion of groundwater level as an predictor variable. 

Overall, models that were trained including groundwater performed worse. While this contradicts the interconnectedness of most groundwater and river systems, this result is most likely tied to my use of historical average values as the predictor variable in the forecast testing. The historical averages were different from the actual values of the predictor variables for the test month, causing a misaligned forecast prediction.

#

### Quick Reference

Review main.py and functions.py for full coding logic.

Hudak_Thesis is the actual paper I submitted to my department.

The results folder includes json, csv, and plot png files. Search these files for the forecasted values, actual data, and perfomance metrics.

The weather_data folder includes pre-downloaded csv files from the NOAA Past Weather portal (https://www.ncei.noaa.gov/access/past-weather/). I did not actually run the model on sites D and E because of their shorter data records.

Stream discharge and groundwater data comes from through the USGS dataretrieval package. Check out their GitHub repo: https://github.com/DOI-USGS/dataretrieval-python

#
### Ideas for future reasearch (feel free to clone and fork)
1. Experiment with different methods for using the predictor variables in the forecast. This includes using the actual predictor variable values, or developing univariate ARIMA models for each predictor variable and inputting those predictor ARIMAs into the main stream discharge ARIMA.
2. Alter training/scoring methods so that models prioritize flood prediciton.
3. Treat model predictions as a simulation problem. That is, experiment with different precipitation and groundwater forecasts to assess the model's reaction.
4. Tie in live weather forecasts as predictor variables.
