# Importing Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

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

# Splitting the data into Training and Testing
data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])

scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i - 100:i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Load Model
model = load_model("Tensorflow_keras_model.h5")

# Testing part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

input_data = scaler.transform(final_df)  # Use transform instead of fit_transform

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)

# Rescaling predicted and actual values back to original scale
scale_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Final Graph
st.subheader('Prediction Vs Original')
fig = plt.figure(figsize=(12, 6))
plt.plot(y_test, label="Original price", color='blue')
plt.plot(y_predicted, label="Predicted price", color='red')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)  # Use st.pyplot instead of plt.show()