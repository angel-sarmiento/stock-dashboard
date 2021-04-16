#%% Importing libraries
import numpy as np
import pandas as pd

import yfinance as yf
import matplotlib.pyplot as plt

import plotly.graph_objects as go

import pycaret
from datetime import date


#%% First plot


def grab_stock(stock_ticker):
    stock = yf.Ticker(stock_ticker)
    
    history = stock.history(period="5y")
    
    column_names = ['Open', 'High', 'Low', 'Close', 'Volume'] 
    
    df = history[history.columns & column_names]
    df.index = pd.DatetimeIndex(df.index).to_series()

    return df

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
    title = selected + " Price"
    )

    return fig

#%% Row 2
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
        
    fig.update_yaxes(
        title = selected + " Price"
    )

    fig.update_xaxes(
        title = "Quarter"
    )

    fig.update_traces(marker_color='#ed145a')

    return fig


def volume_bar(df):

    fig = go.Figure(data= [go.Bar(y = df['Volume'], x = df.index)]
    )

    fig.update_layout(
        title = selected.upper() + " Volume",
        width = 600,
        height = 600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
        )
    
    fig.update_yaxes(
        title = selected + " Volume"
    )

    fig.update_xaxes(
        title = "Date"
    )

    fig.update_traces(marker_color='#22abc7')

    return fig