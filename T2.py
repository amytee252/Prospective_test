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

df = pd.read_csv("./prospective-ds-home-challenge/datasets/data_observations.csv")
print(df.info())
print(df)
df.isna().sum() #Detect empty cells in the data
df = df.dropna() #remove rows containing empty cells
print(df)


to_drop = ['date' ]  #List of columns of data to drop from dataframe as they are irrelevant for training
df.drop(to_drop, inplace=True, axis=1) #Remove the columns of data
print(df)

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

#sns.pairplot(train_dataset[['stop_id', 'wet_weather_score', 'boardings', 'alightings', 'dwell_time']], diag_kind='kde')



def df_to_dataset(dataframe,  batch_size=1):  #Function to convert the dataframe to a tensorflow dataframe that can be ML'd on. Temporarily remove the target column
	dataframe = dataframe.copy()
	new_df = dataframe.pop("dwell_time")
	new_df =tf.convert_to_tensor(new_df)
	return new_df


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

model.compile( loss = tf.keras.losses.mae, #mae stands for mean absolute error
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

def plot_loss(x , y):
	plt.figure
	plt.plot(history.history[x])  #Make a plot of the mae (similar to loss) vs. epochs
	plt.plot(history.history[y])
	plt.title('model mse')
	plt.ylabel('mse')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.draw()
	plt.savefig('plot.png')
	plt.figure().clear()
	plt.close()
	plt.cla()
	plt.clf()

plot_loss(x,y)

x_p=test_ds
y_p=prediction
def plot_train_test(x_p, y_p):	#Plot the predictions vs. true values with a scatter of the 'residuals'
	plt.figure
	plt.scatter(x_p, y_p, c='crimson', s =2)
	p1 = max(max(y_p), max(x_p))
	p2 = min(min(y_p), min(x_p))
	plt.plot(['test','predictions'], [p1,p2], 'b-')
	plt.xlabel('test')
	plt.ylabel('predictions')
	plt.axis('equal')
	plt.legend([x_p, y_p], loc='upper left')
	plt.draw()
	plt.savefig('plot2.png')
	plt.figure().clear()
	plt.close()
	plt.cla()
	plt.clf()

plot_train_test(x_p,y_p)














