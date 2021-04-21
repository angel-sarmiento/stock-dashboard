#%%
import numpy as np 
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

import datetime
from datetime import date

import plotly.express as px

today = date.today()

t0 = today.strftime("%Y-%m-%d")

#%% 
#import pycaret
on = yf.Ticker("ON")

#%%
history = on.history(period="10y")

stock_price = history['Open']

stock = pd.DataFrame(stock_price)
# %%
import pycaret 
from pycaret.regression import *
# %%

def get_time(data):
    # extract month and year from dates
    data['Month'] = data.index.month
    data['Year'] = data.index.year
    data['Day'] = data.index.day
    # create a sequence of numbers
    data['Series'] = np.arange(1,len(data)+1)
    
    return data

time_data = get_time(stock)


# %%


train = time_data[(time_data['Year'] <= today.year) & (time_data['Month'] < today.month)]
test = time_data[(time_data['Year'] == today.year) & (time_data['Month'] == today.month)]

#%%
# initialize setup
s = setup(data = train, test_data = test, target = 'Open', fold_strategy = 'timeseries', numeric_features = ['Day', 'Series', 'Year'], fold = 5, transform_target = True, session_id = 123, silent = True, verbose = False)

# %%
best = compare_models(sort = 'MAE')

# %%
# generate predictions on the original dataset
predictions = predict_model(best, data=time_data)
# add a date column in the dataset
#predictions['Date'] = pd.date_range(start='2000-05-02', end = '2021-04-16', freq = 'MS')
# line plot
fig = px.line(predictions, x=predictions.index, y=["Open", "Label"], template = 'plotly_dark')
# add a vertical rectange for test-set separation
fig.add_vrect(x0="2021-04-01", x1=t0, fillcolor="grey", opacity=0.25, line_width=0)
fig.show()

#%%
tmrw = today + datetime.timedelta(days = 1) 
tmrw_f = tmrw.strftime("%Y-%m-%d")
mth = today + datetime.timedelta(days = 30)
mth_f = mth.strftime("%Y-%m-%d")

future_dates = pd.date_range(start = tmrw_f, end = mth_f, freq = 'D')
future_df = pd.DataFrame()
future_df['Month'] = [i.month for i in future_dates]
future_df['Year'] = [i.year for i in future_dates]    
future_df['Day'] = [i.day for i in future_dates]
future_df['Series'] = np.arange(len(time_data),(len(time_data) +len(future_dates)))

#%%
future_df_i = future_df.set_index(pd.date_range(start=tmrw_f, end=mth_f, freq='D'))
conc = pd.concat([time_data, future_df_i])


#%%
fig = px.line(conc, x=conc.index y=["Open", "Label"], template = 'plotly_dark')
fig.show()

# %%
save_model(best, 'deployment_model')
# %%
test = on.options
# %%
on.get_info()['longBusinessSummary']
# %%
