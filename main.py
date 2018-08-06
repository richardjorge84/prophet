# Final CS50x Project. Jorge Richard Angulo/2018.
# Pandas is imported to get data from 'history'.xlsx Sheet1 file
import pandas as pd
# Prophet to be the FB API to call
from fbprophet import Prophet
# matplotlib to generate output chart to see the fit, forecasts and bounds
import matplotlib.pyplot as plt
# Numpy to select in data matrix the original values to avoid plot "ceros"
import numpy as np


def main():

    # Get the data from excel file called 'history.xlsx' on Sheet 1; cell A1=ds and A2=y
    data = pd.read_excel('history.xlsx', sheet_name='Sheet1')
    # Calling the Prophet model. See other seasonalities in FB Prophet documentation
    model = Prophet(yearly_seasonality=25, weekly_seasonality=25)
    model.fit(data)
    # Cutomize number of days if needed
    num_days = 180
    future = model.make_future_dataframe(periods=num_days)
    forecast = model.predict(future)
    data.set_index('ds', inplace=True)
    forecast.set_index('ds', inplace=True)
    full_matrix = data.join(forecast[['yhat', 'yhat_lower', 'yhat_upper']], how='outer')
    # Change "M" (for months) to "W" for weeks or "D" for days
    full_matrix = full_matrix.resample("M").sum()
    original_value_count = np.count_nonzero(full_matrix['y'])
    # Chart names can be customized in this section
    full_matrix['Original (y)'] = full_matrix['y'][0:original_value_count]
    full_matrix['Forecasts'] = full_matrix['yhat']
    full_matrix['LCL'] = full_matrix['yhat_lower']
    full_matrix['UCL'] = full_matrix['yhat_upper']
    full_matrix[['Original (y)', 'Forecasts', 'LCL', 'UCL']].plot()
    plt.legend(loc=0)
    # This will save a png file to see the fit, forecast and bounds
    plt.savefig('prophet.png', dpi=200, bbox_inches='tight')
    # The full excel file with forecast and bounds will be provided, gropued as per line 25 value
    writer = pd.ExcelWriter('forecasts.xlsx')
    full_matrix[['Original (y)', 'Forecasts', 'LCL', 'UCL']].to_excel(writer, 'Sheet1')
    writer.save()


if __name__ == "__main__":
    main()
