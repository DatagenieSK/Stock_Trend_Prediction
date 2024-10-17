#Importing Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from keras.models import load_model
import streamlit as st


start = '2010-01-01'
end = '2024-10-18'

st.title('Stock Trend Prediction')
user_input = st.text_input('Enter Stock Ticker', 'TSLA')
df = yf.download(user_input, start=start, end=end)

#describing data 

st.subheader('Data From 2010 to 18/10/2024')
st.write(df.describe())