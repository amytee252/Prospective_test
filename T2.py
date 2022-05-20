##Script to carry out task 2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import feature_column


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from Functions import *
from Plots import *

df1 = pd.read_csv("./prospective-ds-home-challenge/datasets/data_observations.csv")
df2 = pd.read_csv("./prospective-ds-home-challenge/datasets/data_stops.csv")

print(df2)
print(df2.info() )

df2['is_urban'] = df2['is_urban'].astype(bool)
df1['is_urban'] = df1['stop_id'].map(df2.set_index('stop_id')['is_urban'])

df = df1

df.isna().sum() #Detect empty cells in the data
df = df.dropna() #remove rows containing empty cells
print(df)


to_drop = ['date' ]  #List of columns of data to drop from dataframe as they are irrelevant for training
df.drop(to_drop, inplace=True, axis=1) #Remove the columns of data
print(df)

for column in df.columns:
	print('Creating plots of ', str(column) , 'verses dwell time') 
	swarm_plot(str(column), 'dwell_time',df)
	scatter_plot(str(column), 'dwell_time', df)

df_copy = df.copy() 

scaler = MinMaxScaler()
df= pd.DataFrame(scaler.fit_transform(df), columns=df.columns)  #Normalizes the data in the dataframe
print(df)
df['dwell_time'] = df_copy['dwell_time']
print(df)
df = df.sample(frac = 1)  #Randomizes the sample

print(df)

df['dwell_time'] = df_copy['dwell_time']
print(df)

train_dataset = df.sample(frac=0.8, random_state=0)
test_dataset = df.drop(train_dataset.index)

batch_size = 1

train_target = train_dataset["dwell_time"] 
test_target = test_dataset["dwell_time"]

train_ds = df_to_dataset(train_dataset, batch_size=batch_size) #create coverted train dataset and select batch size

test_ds = df_to_dataset(test_dataset,  batch_size=batch_size)



tf.random.set_seed(42) 
print('number of columns: ', len(train_dataset))

model = tf.keras.Sequential([  #build NN
			tf.keras.layers.Dense(len(train_dataset) , activation='relu'),
			tf.keras.layers.Dense(len(train_dataset) , activation='relu'),
			tf.keras.layers.Dense(1)
])

model.compile( loss = tf.keras.losses.mse, #mae stands for mean squared error
              optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001), #stochastic GD
              metrics = ['mse'])

history = model.fit( train_ds, train_target, validation_data=(test_ds, test_target), epochs = 100)  #Do the training

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


prediction = model.predict(test_ds)  #Make predictions on the testing data
print('predictions')
print(np.round(prediction, 6))

print(model.summary() )

mse = tf.metrics.mean_squared_error( test_target, prediction) #Calculate mse
print('MSE')
print(mse)

print(history.history.keys())
x = 'mse' 
y = 'val_mse'
plot_loss(x,y)

x_p=test_ds
y_p=prediction
plot_train_test(x_p,y_p)













