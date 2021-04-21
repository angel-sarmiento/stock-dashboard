#!/usr/local/bin/python3.8
#%% Importing libraries
import streamlit as st
import statsmodels as sm
import numpy as np
import pandas as pd

import yfinance as yf
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.express as px

import pycaret
from pycaret.regression import *
from datetime import date, timedelta

import random


from streamlit import caching
caching.clear_cache()

today = date.today()

t0 = today.strftime("%Y-%m-%d")

# getting time periods for tomorrow and a month after
tmrw = today + timedelta(days = 1) 
tmrw_f = tmrw.strftime("%Y-%m-%d")
mth = today + timedelta(days = 30)
mth_f = mth.strftime("%Y-%m-%d")

# Changing the dashboard to the wide layout 
st.set_page_config(layout="wide")
#%%  Search button and Introduction
row1_1, row1_2 = st.beta_columns((2,3))

#app preamble and setup for the search bar
with row1_1:
    st.title("We Like the Stock")
    st.subheader("A Web App by [Angel Sarmiento](https://github.com/angel-sarmiento)")

    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            
    def remote_css(url):
        st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)    

    def icon(icon_name):
        st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)

    local_css("style.css")
    remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')
    #icon("search")
    selected = st.text_input("", "Type a Stock Ticker...", help='Just the ticker, for example: AAPL, aapl, msft. ')
    #button_clicked = st.button("Enter")
    range_sel = st.selectbox(
            'Select a time range:', 
            ('1y',
            '5y',
            '10y',
            'Max'), 
            help="""This may break the predictions that are done at the bottom of this dashboard. 
            Ideally you would want a larger range of time values, however some stocks will still not 
            work despite this."""
            )

with row1_2:
    st.write(
    """
    ##
    The purpose of this dashboard is to develop a broad sense of a stock's potential by using both the Yahoo Finance api
    to track the stocks themselves and the FRED api to track the general state of the economy. There is also some prediction
    done using AutoML and rapid prototyping of machine learning models using pycaret. The Information should not be construed 
    as investment/trading advice and is not meant to be a solicitation or recommendation to buy, sell, or hold any securities 
    mentioned. Dislamer: This does not work for certain stocks and does not work for ETFs and Cryptocurrency. Namely, everything 
    other than prices for these securities will result in an error.  
    """)

#%% First plot

def general_stock(selected):
    stock = yf.Ticker(selected)
    return stock

gen_stock = general_stock(selected)

main_head_str = "Visualizing the Stock Price of " + gen_stock.get_info()['longName'] 



st.header(main_head_str)


#First plot

# getting the stock and its important features
def grab_stock(stock_ticker, range_sel):
    #stock = yf.Ticker(stock_ticker)
    
    history = gen_stock.history(period=range_sel)
    
    column_names = ['Open', 'High', 'Low', 'Close', 'Volume'] 
    
    df = history[history.columns & column_names]
    df.index = pd.DatetimeIndex(df.index).to_series()

    return df

#tables showing information like financials, option chains, institutional holders
def stock_tables(stock_ticker):
    stock = yf.Ticker(stock_ticker) 
    
    unwanted_cols = ['contractSymbol', 'lastTradeDate', 'contractSize', 'currency']

    # All of the necessary to view tables from yfinance
    q_fin = pd.DataFrame(stock.quarterly_financials)
    recommendations = pd.DataFrame(stock.recommendations)
    y_fin = pd.DataFrame(stock.financials)
    in_holders = pd.DataFrame(stock.institutional_holders)
    maj_holders = pd.DataFrame(stock.major_holders)
    calendar = pd.DataFrame(stock.get_calendar())
    q_earn = pd.DataFrame(stock.quarterly_earnings)
    y_earn = pd.DataFrame(stock.earnings)
    calls = pd.DataFrame(stock.option_chain().calls).drop(unwanted_cols, axis = 1)
    puts = pd.DataFrame(stock.option_chain().puts).drop(unwanted_cols, axis = 1)

    return q_fin, recommendations, y_fin, in_holders, maj_holders, calendar, q_earn, y_earn, calls, puts


def display_candlestick(df):
    """ This is the main plotting method for showing off the stock 
    """
    fig = go.Figure(data=[go.Candlestick(x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'])]
        )
        
    fig.update_layout(
        title = selected.upper() + " Stock Prices",
        width = 1700,
        height = 600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
        )

    fig.update_traces(increasing_line_color= 'cyan', decreasing_line_color= '#f890e7')
        
    fig.update_yaxes(
    title = "Price"
    )

    return fig


#%% Main function calls for row 1

if selected != "":  
    df = grab_stock(selected, range_sel)
    fig = display_candlestick(df)
    st.plotly_chart(fig)
    st.header('About the Company')
    gen_stock.get_info()['longBusinessSummary']
# running the funtion above to get the stock information
q_fin, recommend, y_fin, in_holders, maj_holders, calendar, q_earn, y_earn, call_op, put_op = stock_tables(selected)


#%% Row 2
# net income bar chart 
def display_bar(df):
    df = df.transpose()
    fig = go.Figure(data=[go.Bar(y=df['Net Income'], x = df.index)])
        
    fig.update_layout(
        title = selected.upper() + " Quarterly Net Income",
        width = 600,
        height = 600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
        )

    fig.update_xaxes(
        title = "Quarter"
    )

    fig.update_traces(marker_color='#f890e7')

    return fig

#bar chart for volume of shares
def volume_bar(df):

    fig = go.Figure(data= [go.Bar(y = df['Volume'], x = df.index)]
    )

    fig.update_layout(
        title =  "Volume of Shares",
        width = 600,
        height = 600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
        )

    fig.update_xaxes(
        title = "Date"
    )

    fig.update_traces(marker_color='#a1b1d0')

    return fig

# bar plot for the institutional holders 
def holder_bar(df):

    fig = go.Figure(data= [go.Bar(y = df['Shares'], x = df['Holder'])]
    )

    fig.update_layout(
        title = "Institutional Shares",
        width = 600,
        height = 600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
        )

    fig.update_traces(marker_color='cyan')

    return fig

#%% Getting the 3rd row of plots
row2_1, row2_2, row2_3 = st.beta_columns((1,1,1))

#%% 

with row2_1:
    fig1 = display_bar(q_fin)
    st.plotly_chart(fig1)

with row2_2:
    fig2 = volume_bar(df)
    st.plotly_chart(fig2)

with row2_3:
    fig3 = holder_bar(in_holders)
    st.plotly_chart(fig3)


#%%

def get_exps(stock_ticker):
    exp = list(gen_stock.options)
    return exp


def get_options(date, stock_ticker, type_of):
    stock = yf.Ticker(stock_ticker) 
    
    unwanted_cols = ['contractSymbol', 'lastTradeDate', 'contractSize', 'currency']
    
    if type_of == 'Calls':
        values = pd.DataFrame(stock.option_chain(date).calls).drop(unwanted_cols, axis = 1)
    elif type_of == 'Puts':
        values = pd.DataFrame(stock.option_chain(date).puts).drop(unwanted_cols, axis = 1)


    #calls = pd.DataFrame(stock.option_chain(date).calls).drop(unwanted_cols, axis = 1)
    #puts = pd.DataFrame(stock.option_chain(date).puts).drop(unwanted_cols, axis = 1)

    return values

#%%
# 2 columns on third row 
row3_1, row3_2 = st.beta_columns((3,3))

# Setting up the second row of information
with row3_1:
    

    #Dictionary mapping values to respective fully written out strings
    table_dict = {
        'Quarterly Financials': q_fin,
        'Buy/Sell Recommendations': recommend,
        'Yearly Financials': y_fin,
        'Institutional Holders': in_holders,
        'Major Holders': maj_holders,
        'Calendar': calendar,
        'Quarterly Earnings': q_earn,
        'Yearly Earnings': y_earn
    }

    # info select for table/dataframe
    info_select = st.selectbox(
        'Select the Data you are interested in for ' + selected + ' Stock', 
        ('Quarterly Financials',
        'Yearly Financials',
        'Institutional Holders',
        'Major Holders',
        'Calendar',
        'Buy/Sell Recommendations',
        'Quarterly Earnings',
        'Yearly Earnings'))

    st.dataframe(table_dict[info_select], width = 800, height = 700)

with row3_2:

    exps = get_exps(selected)

    #option_dict = {
    #    'Calls': True,
    #    'Puts': False
    #}


    option_select = st.selectbox(
        'Select the Option Chain type for ' + selected + ' Stock:',
        ('Calls',
        'Puts')
    )

    expiry_select = st.selectbox(
        "",
        (
            exps
        )
    )

    option_chain = get_options(date = expiry_select, stock_ticker = selected, type_of = option_select)

    st.dataframe(option_chain, width = 800, height = 579)
    #st.dataframe(option_dict[option_select], width = 800, height = 579)


#%% SECTION FOR THE AUTOML BIT

st.header("Predictions: Where is This Stock Going?")
st.text("""
    To be completely clear, despite the naming of this section, it is pretty much impossible 
    to predict the price of stocks. Keep in mind that when viewing 
    these data. These 'predictions' are more for a general understanding of some likely 
    movements of the underlying security. The model that worked best in offline testing
    was the AdaBoost Regression Model. None of this is investment advice. Drag on the 
    graph below to view a specific section. 
    """)


def get_time_setup(ticker, range_sel):
    #import pycaret
    stock = yf.Ticker(ticker)
    history = stock.history(period=range_sel)
    stock_price = history['Open']

    data = pd.DataFrame(stock_price)

    # extract month and year from dates
    data['Month'] = data.index.month
    data['Year'] = data.index.year
    data['Day'] = data.index.day

    # create a sequence of numbers
    data['Series'] = np.arange(1,len(data)+1)
    
    # Future Dates for predicting out of sample
    future_dates = pd.date_range(start = tmrw_f, end = mth_f, freq = 'B')
    future_df = pd.DataFrame()
    future_df['Month'] = [i.month for i in future_dates]
    future_df['Year'] = [i.year for i in future_dates]    
    future_df['Day'] = [i.day for i in future_dates]
    future_df['Series'] = np.arange(len(data),(len(data) +len(future_dates)))


    return data, future_df

def predict(model, future_df):
    predictions_df = predict_model(model, data=future_df)
    predictions = predictions_df['Label'][0]
    return predictions

time_data, future_df = get_time_setup(selected, range_sel)

time_data.fillna(df.mean(), inplace=True)

#%%
row4_1, row4_2 = st.beta_columns((1,1))



# %%

train = time_data[(time_data['Year'] <= today.year) & (time_data['Month'] < today.month)]
test = time_data[(time_data['Year'] == today.year) & (time_data['Month'] == today.month)]

with row4_1: 

    with np.errstate(divide='ignore'):
        st.subheader("[AdaBoost Regression](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html) Model Metrics (10-Fold Cross-Validation)")
        s = setup(data = train, test_data = test, target = 'Open', fold_strategy = 'timeseries', numeric_features = ['Day', 'Series', 'Year'], transform_target = True, silent = True, verbose = False, transform_target_method='yeo-johnson')
        set_config('seed', random.randint(1, 999))
        # Compare models
        best = create_model('ada', fold = 10)
        tuned_best = tune_model(best, optimize='MSE')    
        model_results = pull()
        st.dataframe(model_results.style.apply(lambda x: ['background: #f890e7' 
                                    if (x.name == 'Mean')
                                    else '' for i in x], axis=1), width = 1000, height = 600)


## TODO: Fix problem where pycaret does not reset the session every time the streamlit dashboard reloads
with row4_2:

    with np.errstate(divide='ignore'):
        # generate predictions on the original dataset
        predictions_future = predict_model(tuned_best, data=future_df)
        # add a date column in the dataset
        # add a vertical rectange for test-set separation

        future_df_i = predictions_future.set_index(pd.date_range(start=tmrw_f, end=mth_f, freq='B'))
        conc = pd.concat([time_data, future_df_i])

        fig5 = px.line(conc, x=conc.index, y=["Open", "Label"], labels = {'variable': ''})

        fig5.update_layout(
            title =  "Predicting the Opening Price",
            width = 800,
            height = 600,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01)
            )


        fig5.update_yaxes(
            title = "Price"
        )
        fig5.update_xaxes(
            title = ""
        )

        st.plotly_chart(fig5)

# %%
