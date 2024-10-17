# Importing Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st

# Define the date range
start = '2010-01-01'
end = '2024-10-18'

# Streamlit app title
st.title('Stock Trend Prediction')

# User input for stock ticker
user_input = st.text_input('Enter Stock Ticker', 'TSLA')

# Fetching the stock data from Yahoo Finance
df = yf.download(user_input, start=start, end=end)

# Displaying the stock data summary
st.subheader('Data From 2010 to 18/10/2024')
st.write(df.describe())

# Visualization of Closing Price
st.subheader('Closing Price Vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Closing Price', color='blue')
plt.title(f'Closing Price of {user_input}')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
st.pyplot(fig)

# Visualization of Closing Price with 100-day Moving Average
st.subheader('Closing Price Vs Time with 100-Day Moving Average')
ma100 = df['Close'].rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Closing Price', color='blue')
plt.plot(ma100, label='100-Day MA', color='orange')
plt.title(f'Closing Price of {user_input} with 100-Day MA')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

# Visualization of Closing Price with 100-day and 200-day Moving Averages
st.subheader('Closing Price Vs Time with 100-Day and 200-Day Moving Averages')
ma200 = df['Close'].rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Closing Price', color='blue')
plt.plot(ma100, label='100-Day MA', color='orange')
plt.plot(ma200, label='200-Day MA', color='green')
plt.title(f'Closing Price of {user_input} with 100-Day and 200-Day MA')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)