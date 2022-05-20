#Script that contains plotting functions across the three main scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import feature_column


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Creates swarm plots feature in the dataset
def swarm_plot(y_var, x_var, dataframe):
	sns_plot = sns.swarmplot(y=dataframe[y_var], x=dataframe[x_var], data=dataframe, s=1, size = 20)
	fig1 = sns_plot.get_figure()
	sns_plot.set_xticklabels(sns_plot.get_xticklabels(),rotation = 90)
	sns_plot.tick_params(axis='x', labelsize=5)
	fig1.savefig('plots/' + y_var + '_' + x_var + '.png', dpi = 300)

	sns_boxplot = sns.boxplot(y=dataframe[y_var], x=dataframe[x_var], data=dataframe, whis=np.inf)
	sns_boxplot.set_xticklabels(sns_plot.get_xticklabels(),rotation = 90)
	sns_boxplot.tick_params(axis='x', labelsize=5)
	fig2 = sns_boxplot.get_figure()
	fig2.savefig('plots/' + y_var + '_' + x_var + '_boxplot.png', dpi = 300)
	



strings = ['stop_id', 'wet_weather_score', 'boardings', 'alightings', 'is_urban']   # array to be used with scatterplot function below

# Creates 2D scatterplots  
def scatter_plot(y_var, x_var, dataframe):
	y_label = y_var
	x_label = x_var + " + " + y_var
	x_label_save = x_var + "+" + y_var
	sns_scatterplot = sns.scatterplot(y=dataframe[y_var], x= dataframe[x_var], data=dataframe, s=2)
	sns_scatterplot.set_xlabel(x_label)
	sns_scatterplot.set_ylabel(y_label)
	handles, labels = sns_scatterplot.get_legend_handles_labels()
	sns_scatterplot.legend()
	fig3 = sns_scatterplot.get_figure()
	fig3.savefig('plots/' + y_label + '_' + x_label_save + '_scatterplot.png', dpi = 300)

def plot_percentage_hourly(day, result):
	plt.figure() 
	plt.title('OTP for ' + str(day))
	names = list(result.keys() )
	percentages = list(result.values() )
	plt.bar(range(len(result)), percentages, tick_label = names)
	plt.ylabel('Percentages(%)')
	plt.xlabel('hour be' + str(day))
	plt.xticks(rotation=90)
	plt.savefig('plots/Hourly_OTP_' + str(day) + '.png')
	plt.figure().clear()
	plt.close()
	plt.cla()
	plt.clf()


def plot_percentage_day(day, result):
	plt.figure() 
	plt.title('OTP for date ' + str(day))
	names = list(result.keys() )
	percentages = list(result.values() )
	plt.bar(range(len(result)), percentages, tick_label = names)
	plt.ylabel('Percentages(%)')
	plt.xlabel('Trip number for date ' + str(day))
	plt.xticks(rotation=90)
	plt.savefig('plots/OTP_day_' + str(day) + '.png')
	plt.figure().clear()
	plt.close()
	plt.cla()
	plt.clf()


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
