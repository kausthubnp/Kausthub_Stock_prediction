#Import required libraries
import numpy as np
import math
import matplotlib.pyplot as plt
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import pandas as pd
plt.rcParams['figure.figsize'] = (8, 3)

def lstm_model(input_dim, output_dim, return_sequences):
 
    
    model = Sequential()
    #input layer
    model.add(LSTM(
        input_shape=(None, input_dim),
        units=output_dim,
        return_sequences=return_sequences))
    #First layer with 10 gates
    model.add(LSTM(
        100,
        return_sequences=True))
    #second layer with 50 gates
    model.add(LSTM(
        50))
    
    #output layer
    model.add(Dense(
        units=1))
    model.add(Activation('linear'))

    return model

from sklearn.preprocessing import MinMaxScaler


def get_normalised_data(data):
  
    # Initializing the scalar functio
    scaler = MinMaxScaler()
    numerical = ['Open', 'Close', 'Volume']
    #numerical = ['Open', 'Close','High','Low','Volume','OHLC','HLC']
    #numerical = ['Open', 'Close','High','Low','Volume']
    data[numerical] = scaler.fit_transform(data[numerical])

    return data


def remove_data(data):
   
    #Initializing variables
    item = []
    open = []
    close = []
    #high = []
    #low = []
    volume = []

   
    i_counter = 0
    for i in range(len(data) - 1, -1, -1):
        #appending the index
        item.append(i_counter)
        open.append(data['Open'][i])
        close.append(data['Close'][i])
        #high.append(data['High'][i])
        #low.append(data['Low'][i])
        volume.append(data['Volume'][i])
        i_counter += 1

    
    stocks = pd.DataFrame()

    # Addition of required factors to data frame
    stocks['Item'] = item
    stocks['Open'] = open
    stocks['Close'] = pd.to_numeric(close)
    #stocks['High'] = pd.to_numeric(high)
    #stocks['Low'] = pd.to_numeric(low)
    stocks['Volume'] = pd.to_numeric(volume)
    #stocks['OHLC'] = data[['Open','High', 'Low', 'Close']].mean(axis = 1)
    #stocks['HLC'] = data[['High', 'Low', 'Close']].mean(axis = 1)

    return stocks

def train_test_split_lstm(stocks, prediction_time=1, test_data_size=450, unroll_length=50):

    # the prediction_time parameter can be used to increase the prediction interval 
    test_data_cut = test_data_size + unroll_length + 1

    # training data
    x_train = stocks[0:-prediction_time - test_data_cut].as_matrix()
    y_train = stocks[prediction_time:-test_data_cut]['Close'].as_matrix()

    # test data
    x_test = stocks[0 - test_data_cut:-prediction_time].as_matrix()
    y_test = stocks[prediction_time - test_data_cut:]['Close'].as_matrix()

    return x_train, x_test, y_train, y_test


def unroll(data, sequence_length=24):

    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    return np.asarray(result)

def plot_lstm_prediction(actual, prediction, title='Tesla Actual vs Prediction', y_label='Price', x_label='Trading Days'):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # Add labels for x axis 
    plt.ylabel(y_label)
    #Add labels for y axis
    plt.xlabel(x_label)
    # Plot actual and predicted close values
    plt.plot(actual, '#00FF00', label='Adjusted Close')
    plt.plot(prediction, '#0000FF', label='Predicted Close')
    # Set title
    ax.set_title(title)
    ax.legend(loc='upper left')
    plt.show()
