import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#Get the Stock Quote
#df= DataFrame
df = web.DataReader('AAPL',data_source='yahoo',start='2012-01-01',end='2019-12-17')
#print(df)

#get the number of roes and colums in dataset

df.shape
#visualizing the closing prce history

# plt.figure(figsize=(16,8))
# plt.title('Close Price History')
# plt.plot(df['Close'])
# plt.xlabel('Date', fontsize=23)
# plt.ylabel('Close Price USD ($)',fontsize=23)
# plt.show()

#creating the new data frame with only the Close column
data =df.filter(['Close'])
#Convert the Dataframe to a numpy array
dataset = data.values
#get the number of rows to train th model on
training_data_len =math.ceil(len(dataset)*.8)  #to train out data on using 80 % of the data in dataset

#print(training_data_len)

#scaling the data (preprocessing the data)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset) #the range is 0 to 1  inclusive i.e both 0 and 1 will also be included
#print(scaled_data)

#creating the training dataset
#1. create the scaled training dataset

train_data = scaled_data[0:training_data_len, :]

#2. split the data into x_train and y_train data sets
x_train = []  #independent variables
y_train = []  #target variable

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    # if i<=60:
    #     print(x_train)
    #     print(y_train)
    #     print()

#convert the x_train an dy_ train in numpy array so we can train them on th LSTM model

x_train, y_train = np.array(x_train), np.array(y_train)

#reshape the x_train dataset, becauus ethe lST model expects the data to be 3Dimensional in the form
#of number of samples,no of time steps and number of featues and right now our x_train dataset in 2dimensional, no of rows and columns
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) #no of sample - number of rows that we have that is 1543,
#print(x_train.shape)

#building the LSTM model

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))


#compiling the model
model.compile(optimizer='adam', loss='mean_squared_error') #optimizier is used to improve upon the loss function
                                                           #loss function is uded to know how good the mdel did while traning

#train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# create the testing dataset

# creating a new array containing scaled value from 1543 to 2003

test_data = scaled_data[training_data_len - 60: , :]
# create the datasets x_test and y_tests

x_test=[]
y_test= dataset[training_data_len: ,:]  # these will be all the values that we wan tour model to predict

for i in range(60 ,len(test_data)):
    x_test.append(test_data[i - 60:i, 0])

# convert the data to numpy array
x_test = np.array(x_test)

# reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# get the models predicted price values

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# we would like to evaluate our model by getting the root mean square error
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))

#plot the data

train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

#visualize the model
plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date', fontsize = 23)
plt.ylabel('Close Price USD ($)', fontsize=23)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
print(valid)
plt.show()

#now we are trying to predict for the date that is not is the dataset i.e 2019-12-18

# Get the quote
apple_quote = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end ='2019-12-17')
#new DF
new_df = apple_quote.filter(['Close'])

# Get the last 60 days closing price values and conver the dataframe to an array

last_60_days = new_df[-60:].values
#scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)
#Create an empty list
X_test = []
#append the past 60 days to the X_test

X_test.append(last_60_days_scaled)

#convert the X_text dataset to a numpy array
X_test = np.array(X_test)
#reshaping the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#Get the predicted scaled price
pred_price = model.predict(X_test)

#undo the scalling
pred_price = scaler.inverse_transform(pred_price)

print('The predicted price is ::' + pred_price)
apple_quote2 = web.DataReader('AAPL', data_source='yahoo', start='2019-12-18', end ='2019-12-18')
print(apple_quote2['Close'])


