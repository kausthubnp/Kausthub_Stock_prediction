#Import the required libraries
import math
import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

import lstm, time 
import preprocess_data as ppd
import visualize as vs
import stock_data as sd
import Data_processiong_modelling as dpm

#Read the data
data = pd.read_csv('tesla1.csv')

#Function to remove unused data
stocks = dpm.remove_data(data)

#Normalize the data
stocks = dpm.get_normalised_data(stocks)
#droping the index
stocks_data = stocks.drop(['Item'], axis =1)
#split into train and test data
X_train, X_test,y_train, y_test = dpm.train_test_split_lstm(stocks_data, 5)

unroll_length = 50
X_train = dpm.unroll(X_train, unroll_length)
X_test = dpm.unroll(X_test, unroll_length)
y_train = y_train[-X_train.shape[0]:]
y_test = y_test[-X_test.shape[0]:]

print("x_train", X_train.shape)
print("y_train", y_train.shape)
print("x_test", X_test.shape)
print("y_test", y_test.shape)

#Passing parameters into the model
model = dpm.lstm_model(input_dim = X_train.shape[-1],output_dim = unroll_length, return_sequences=True)
#print (input_dim)

# Compile the model
start = time.time()
#compute the mean squared error
model.compile(loss='mean_squared_error', optimizer='adam')
print('compilation time : ', time.time() - start)

#Fit the LSTM model : Parameters
model.fit(
    X_train,
    y_train,
    epochs=20,
    validation_split=0.05)

#predict for test and train data
predictions_test = model.predict(X_test)
predictions_train = model.predict(X_train)

#Visualize the plots
dpm.plot_lstm_prediction(y_test,predictions_test)
df1 = pd.DataFrame(y_test)
df2 = pd.DataFrame(predictions_test)

#df3 = pd.DataFrame(denorma_y_test)
df = pd.concat([df1,df2],axis=1)
df.to_csv('tesla_final.csv')

