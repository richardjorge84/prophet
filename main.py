# Implementation of FB Prophet for Time series prediction. Jorge Richard/2018.
# Pandas is imported
import pandas as pd
# Calling Prophet to create the model 
from fbprophet import Prophet
# Call plots (based on matplotlib) to plot outcomes
from fbprophet.plot import add_changepoints_to_plot

# Call the data from *.csv file. 
# The input to Prophet is always a dataframe with two columns: ds and y. 
# The ds (datestamp) column should be of a format expected by Pandas, ideally YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp. 
# The y column must be numeric, and represents the measurement we wish to forecast.
df = pd.read_csv('library.csv')
# Validate the data
print(df.head())
# create the variable to hold the data
m = Prophet()
# fit over m the data pulled from csv
m.fit(df)
# Predict for the selected range (periods stated here as days. See documentation of prophet for other type of periods)
future = m.make_future_dataframe(periods=180)
print(future.tail())
# Save on variable forecast the predictions
forecast = m.predict(future)
# Select the columns to visualize prelim, ds = date, yhat = forecast, yhat_lower = lower limit, yhat_upper = upper limit
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
# Plot the forecast o fig1. Then add the main trend changes.
fig1 = m.plot(forecast)
a = add_changepoints_to_plot(fig1.gca(), m, forecast)
# Plot the components of the prediction, trend, seasonality by YYYY, MM, Week and DD.
fig2 = m.plot_components(forecast)
# Write the output predictions as csv
forecast.to_csv("forecastout.csv")

