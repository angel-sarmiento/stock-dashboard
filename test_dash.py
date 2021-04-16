#!/usr/local/bin/python3.8
#%% Importing libraries
import streamlit as st
import statsmodels as sm
import numpy as np
import pandas as pd

import yfinance as yf
import matplotlib.pyplot as plt

import plotly.graph_objects as go

import pycaret
from datetime import date

today = date.today()

t0 = today.strftime("%Y-%m-%d")

# Changing the dashboard to the wide layout 
st.set_page_config(layout="wide")
#%%  Search button and Introduction
row1_1, row1_2 = st.beta_columns((2,3))

#app preamble and setup for the search bar
with row1_1:
    st.title("Track Your Favorite Stocks")
    st.subheader("A web app by [Angel Sarmiento](https://github.com/angel-sarmiento)")

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
    selected = st.text_input("", "Type a Stock Ticker...")
    #button_clicked = st.button("Enter")

with row1_2:
    st.write(
    """
    ##
    The purpose of this dashboard is to develop a broad sense of a stock's potential by using both the Yahoo Finance api
    to track the stocks themselves and the FRED api to track the general state of the economy. There is also some prediction
    done using AutoML and rapid prototyping of machine learning models using pycaret. Note that this does not
    constitute financial advice and I am not a CPA nor a financial professional. 
    """)

#%% First plot

# getting the stock and its important features
def grab_stock(stock_ticker):
    stock = yf.Ticker(stock_ticker)
    
    history = stock.history(period="5y")
    
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
    df = grab_stock(selected)
    fig = display_candlestick(df)
    st.plotly_chart(fig)

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

    option_dict = {
        'Calls': call_op,
        'Puts': put_op
    }

    option_select = st.selectbox(
        'Select the Option Chain type for ' + selected + ' Stock:',
        ('Calls',
        'Puts')
    )

    st.dataframe(option_dict[option_select], width = 800, height = 579)


