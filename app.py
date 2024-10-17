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

#Visulizations
st.subheader('Closing Price Vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)