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
from sklearn.metrics import roc_curve,roc_auc_score

from Functions import *
from Plots import *

df1 = pd.read_csv("./prospective-ds-home-challenge/datasets/data_observations.csv")
df2 = pd.read_csv("./prospective-ds-home-challenge/datasets/data_stops.csv")

print(df2)
print(df2.info() )


#IF I had time, I would have tried to predict whether a stop is urban or not, so I add it to the dataset, but later drop it off, so it is not included in the training.
df2['is_urban'] = df2['is_urban'].astype(bool)
df1['is_urban'] = df1['stop_id'].map(df2.set_index('stop_id')['is_urban'])

df = df1

df.isna().sum() #Detect empty cells in the data
df = df.dropna() #remove rows containing empty cells
print(df)



to_drop = ['date' ]  #List of columns of data to drop from dataframe as they are irrelevant for training
df.drop(to_drop, inplace=True, axis=1) #Remove the columns of data
print(df)


#Make some scatter plots of variables to see correlations
for column in df.columns:
	print('Creating plot of ', str(column) , 'verses dwell time') 
	scatter_plot(str(column), 'dwell_time', df)

df_copy = df.copy() 

scaler = MinMaxScaler() 
df= pd.DataFrame(scaler.fit_transform(df), columns=df.columns)  #Normalizes the data in the dataframe
#print(df)
df['dwell_time'] = df_copy['dwell_time'] #Don't want to normalise dwell_time so that the original data in the copy of the original df.
#print(df)
df = df.sample(frac = 1)  #Randomizes the sample

print(df)

train_dataset = df.sample(frac=0.8, random_state=0)  #Select 80% of events for training
test_dataset = df.drop(train_dataset.index) #Remaining events for testing

batch_size = 1

train_target = train_dataset["dwell_time"] #Select the variable for the target
test_target = test_dataset["dwell_time"]

train_ds = df_to_dataset(train_dataset, batch_size=batch_size) #create coverted train dataset and select batch size

test_ds = df_to_dataset(test_dataset,  batch_size=batch_size)

tf.random.set_seed(42) 
print('number of columns: ', len(df.columns)) #number of columns in original dataframe

model = tf.keras.Sequential([  #build DNN
			tf.keras.layers.Dense(len(df.columns) -2, activation='relu'),  #input nodes is same as number of training features (in this case 4)
			tf.keras.layers.Dense(len(df.columns) -2, activation='relu'),
			tf.keras.layers.Dense(1)
])

model.compile( loss = tf.keras.losses.mse,  #mse
              optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001), #Adam
              metrics = ['mse'])

history = model.fit( train_ds, train_target, validation_data=(test_ds, test_target), epochs = 100)  #Do the training

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


prediction = model.predict(test_ds)  #Make predictions on the testing data to predict the dwell time
print('predictions')
print(prediction)


mse = tf.metrics.mean_squared_error( test_target, prediction) #Calculate mse
print('mse')
print(mse)


print(history.history.keys())
x = 'mse' 
y = 'val_mse'

#make a plot of the loss per epoch for training and testing 
def plot_loss(x , y):
	plt.figure() 
	plt.plot(history.history[x])  #Make a plot of the mse (similar to loss) vs. epochs
	plt.plot(history.history[y])
	plt.title('model mse')
	plt.ylabel('mse')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	#plt.show()
	plt.draw()
	plt.savefig('plot_loss.png')
	plt.figure().clear()
	plt.close()
	plt.cla()
	plt.clf()

plot_loss(x,y)

#print(test_target)
#print(prediction)

#Plot a scatter plot of the true vs. predicted values. If they are 100% the same they will fall on the diagonal (blue line)
x_p=test_target
y_p=prediction
def plot_train_test(x_p, y_p):	#Plot the predictions vs. true values
	plt.figure
	plt.scatter(x_p, y_p, c='crimson', s =2)

	p1 = max(max(y_p), max(x_p))
	p2 = min(min(y_p), min(x_p))
	plt.plot([p1,p2], [p1,p2], 'b-')
	plt.xlabel('true value')
	plt.ylabel('predictions')
	plt.axis('equal')
	plt.draw()
	plt.savefig('plot_predicted_vs_true.png')
	plt.figure().clear()
	plt.close()
	plt.cla()
	plt.clf()

plot_train_test(x_p,y_p)


#Plot all the predicted and true dwell_time values
def plotGraph(y_test,y_pred):
    if max(y_test) >= max(y_pred):
        my_range = int(max(y_test))
    else:
        my_range = int(max(y_pred))
    plt.scatter(range(len(y_test)), y_test, color='blue')
    plt.scatter(range(len(y_pred)), y_pred, color='red')
    plt.legend(['test', 'predictions'], loc='upper left')
    plt.draw()
    plt.savefig('plot_values_test_pred.png')
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()



plotGraph(x_p, y_p)
