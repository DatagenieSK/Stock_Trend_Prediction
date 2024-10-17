# Stock Trend Prediction App

This repository contains the code for a stock trend prediction application built using **Streamlit**, **Keras**, and **Yahoo Finance API**. The app allows users to visualize stock trends and predict future prices using a pre-trained deep learning model.

### [Live Demo of the App](https://datageniesk-stock-trend-prediction-app-ei2rt0.streamlit.app/)

## Features

- **User Input**: Users can input any stock ticker symbol (e.g., TSLA) to analyze stock data.
- **Historical Data Visualization**: The app displays the stock's historical closing price along with its 100-day and 200-day moving averages.
- **Machine Learning Prediction**: The app predicts future stock prices based on historical data using a pre-trained deep learning model.
- **Comparison Chart**: The app plots predicted stock prices versus the actual prices for easy comparison.

## Installation

To run the application locally, follow the steps below:

1. Clone this repository:
   ```bash
   git clone https://github.com/DatagenieSK/Stock_Trend_Prediction.git
   cd stock-trend-prediction-app
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: .\env\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## How It Works

- The app uses **Yahoo Finance API** to download stock data.
- It preprocesses the data by calculating moving averages and normalizing the prices.
- A pre-trained **Keras** deep learning model is used to predict future prices.
- The predicted prices are compared against actual prices to show model accuracy.

## Usage

1. Navigate to the [live app](https://datageniesk-stock-trend-prediction-app-ei2rt0.streamlit.app/).
2. Enter a stock ticker (e.g., `TSLA` for Tesla).
3. Visualize historical stock prices and moving averages.
4. View the predicted stock prices and compare them with actual prices.

## File Structure

- `app.py`: The main Streamlit app script.
- `Tensorflow_keras_model.h5`: The pre-trained Keras model for stock price prediction.
- `requirements.txt`: A list of dependencies required to run the app.

## Dependencies

The app requires the following Python libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `yfinance`
- `keras`
- `scikit-learn`
- `streamlit`

## Pre-trained Model

The pre-trained model (`Tensorflow_keras_model.h5`) is used to predict stock prices. You can replace this model with your own custom-trained model if needed.

## License

This project is licensed under the MIT License.