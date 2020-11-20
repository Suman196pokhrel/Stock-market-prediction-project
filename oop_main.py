import math
import pandas_datareader as web

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt


class MainLogic:
    def __init__(self, stockName='AAPL', dataSource='yahoo'):
        self.stockName = stockName
        self.dataSource = dataSource
        self.data = None
        self.dataset = None
        self.training_data_len = None
        self.scalar = None
        self.scaled_data = None
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.train_data = None
        self.model = None

    def get_stock_quote(self):
        # Get The Stock Quote
        self.df = web.DataReader(self.stockName, data_source=self.dataSource, start='2012-01-01', end='2019-12-17')
        print(self.df.head())

    def preprocessing(self, scale_start=0, scale_end=1):
        # Only taking Closing Price in the DF
        self.data = self.df.filter(['Close'])

        # convert the Dataframe to a numpy array
        self.dataset = self.data.values

        # compute the number of rows to train the model
        self.training_data_len = math.ceil(len(self.dataset) * 0.8)
        print("Length of Our Training Dataset ==>> ", self.training_data_len)

        # Scaling the data
        self.scalar = MinMaxScaler(feature_range=(scale_start, scale_end))

        # Scaled Dataset
        self.scaled_data = self.scalar.fit_transform(self.dataset)

        # Create The training Dataset
        # 1. create the scaled training dataset
        self.train_data = self.scaled_data[0:self.training_data_len, :]

        # 2. Split the data in train test
        for i in range(60, len(self.train_data)):
            self.x_train.append(self.train_data[i - 60:i, 0])
            self.y_train.append(self.train_data[i, 0])

        # Converting x_train, y_train to numpy array ,
        # because LSTM only works on NumPy array input
        self.x_train, self.y_train = np.array(self.x_train), np.array(self.y_train)

        # Reshape the X_train Dataset
        # because LSTM expects the data
        # to be in 3-Dimension(no of samples, no of steps, no of features)
        # but right now our data is 2D(no of rows , no of columns)
        self.x_train = np.reshape(self.x_train, (self.x_train.shape[0], self.x_train.shape[1], 1))

    def build_lstm_model(self, neurons_LSTM=50, neurons_dense=25):
        self.model = Sequential()
        self.model.add(LSTM(neurons_LSTM, return_sequences=True, input_shape=(self.x_train.shape[1], 1)))
        self.model.add(LSTM(neurons_LSTM, return_sequences=False))
        self.model.add(Dense(neurons_dense))
        self.model.add(Dense(1))

    def model_compile(self, number_of_epochs=1):
        self.model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        self.model.fit(self.x_train, self.y_train, batch_size=1, epochs=number_of_epochs)

    def create_test_dataset(self):
        # Create a new array containing scale value from 1543 to 2003
        self.test_data = self.scaled_data[self.training_data_len - 60:, :]

        # create the datasets
        self.y_test = self.dataset[self.training_data_len:, :]
        for i in range(60, len(self.test_data)):
            self.x_test.append(self.test_data[i - 60:i, 0])

        # converting it to a numpy array
        self.x_test = np.array(self.x_test)

        # Reshaping the data
        self.x_test = np.reshape(self.x_test, (self.x_test.shape[0], self.x_test.shape[1], 1))

    def get_predicted_value(self):
        self.predictions = self.model.predict(self.x_test)
        self.predictions = self.scalar.inverse_transform(self.predictions)

    def evaluate_the_model(self):
        # Using root mean squared error for loss(RMSE)
        self.rmse = np.sqrt(np.mean(self.predictions - self.y_test) ** 2)
        print('RMSE Score =>> ', self.rmse)

    def plot_data(self):
        self.train = self.data[:self.training_data_len]
        self.valid = self.data[self.training_data_len:]
        self.valid['Predictions'] = self.predictions

        plt.figure(figsize=(16, 8))
        plt.title('MODEL')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD ', fontsize=18)

        plt.plot(self.train['Close'])
        plt.plot(self.valid[['Close', 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
        plt.show()


obj = MainLogic()
obj.get_stock_quote()
obj.preprocessing()
obj.build_lstm_model()
obj.model_compile()
obj.create_test_dataset()
obj.get_predicted_value()
obj.evaluate_the_model()
obj.plot_data()
