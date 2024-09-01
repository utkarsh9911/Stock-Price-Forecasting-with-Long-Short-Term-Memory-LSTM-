# Stock-Price-Forecasting-with-Long-Short-Term-Memory-LSTM
## Overview

This project aims to predict future stock prices using Long Short-Term Memory (LSTM) networks. By leveraging historical stock data, the model forecasts the closing prices of stocks, providing insights and aiding in decision-making for investors and traders.

## Table of Contents
- [Project Motivation](#project-motivation)
- [Data Description](#data-description)
- [Model Architecture](#model-architecture)
- [Installation](#installation)


## Project Motivation

The stock market is highly dynamic, and accurate prediction of stock prices is a challenging yet crucial task for investors. This project uses LSTM, a type of Recurrent Neural Network (RNN) that is particularly effective for time series forecasting, to predict stock prices based on historical data. The project demonstrates the application of deep learning in finance, aiming to create a robust model that can aid in making informed investment decisions.

## Data Description

The project utilizes historical stock data sourced from [Yahoo Finance](https://finance.yahoo.com/), fetched using the `yfinance` library. The key features used in this project include:

- **Open**: Price at the start of the trading day.
- **High**: Highest price during the trading day.
- **Low**: Lowest price during the trading day.
- **Close**: Price at the end of the trading day.
- **Adj Close**: Adjusted closing price after accounting for corporate actions.
- **Volume**: Number of shares traded during the day.

The model primarily focuses on predicting the **Close** price.

## Model Architecture

The model is built using a Sequential LSTM network, which is trained on the past 60 days of stock price data to predict the price on the next day. The architecture includes:

- **LSTM Layers**: Capture the temporal dependencies in the stock price data.
- **Dense Layers**: Map the LSTM outputs to the final prediction.
- **Loss Function**: Mean Squared Error (MSE) is used to measure the performance of the model.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/stock-price-prediction-lstm.git
