import streamlit as st
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model

import yfinance as yf 


st.title("Stock Price Predictor App")
stock = st.text_input("Enter the stock ID", "TSLA")

from datetime import datetime
end = datetime.now()
start = datetime(end.year-20, end.month, end.day)

tsla_data = yf.download(stock, start, end)

model = load_model("Stock_price_model.keras")
st.subheader('Stock Data')
st.write(tsla_data)

splitting_len = int(len(tsla_data)*0.7)

x_test = pd.DataFrame(tsla_data.Close[splitting_len:])

def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values)
    plt.plot(full_data.Close, 'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig




# Plotting the graphs
st.subheader("Original Close Price and MA for 100 days")
tsla_data["MA_for_100_days"] = tsla_data.Close.rolling(100).mean()
st.pyplot(plot_graph((15,6), tsla_data['MA_for_100_days'], tsla_data, 0))

st.subheader("Original Close Price and MA for 50 days")
tsla_data["MA_for_50_days"] = tsla_data.Close.rolling(50).mean()
st.pyplot(plot_graph((15,6), tsla_data['MA_for_50_days'], tsla_data, 0))

st.subheader("Original Close Price and MA for 20 days")
tsla_data["MA_for_20_days"] = tsla_data.Close.rolling(20).mean()
st.pyplot(plot_graph((15,6), tsla_data['MA_for_20_days'], tsla_data, 0))

st.subheader("Original Close Price and MA for 100 days vs 50 days")
tsla_data["MA_for_100_days"] = tsla_data.Close.rolling(100).mean()
st.pyplot(plot_graph((15,6), tsla_data['MA_for_100_days'], tsla_data, 1, tsla_data['MA_for_50_days']))


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test[['Close']])

x_data = []
y_data = []

for i in range(20, len(scaled_data)):
    x_data.append(scaled_data[i-20:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)
predictions = model.predict(x_data)

inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

ploting_data = pd.DataFrame(
 {
  'original_test_data': inv_y_test.reshape(-1),
    'predictions': inv_pre.reshape(-1)
 } ,
    index = tsla_data.index[splitting_len+20:]
)
st.subheader("Original values vs Predicted values")
st.write(ploting_data)

st.subheader('Original Close Price vs Predicted Close Price')
fig = plt.figure(figsize=(15,6))
plt.plot(pd.concat([tsla_data.Close[:splitting_len+20], ploting_data],axis=0))
plt.legend(["Data not used", "Original Test Data","Predicted Test Data"])

st.pyplot(fig)