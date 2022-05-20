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
from sklearn.metrics import roc_curve,roc_auc_score

strings = ['stop_id', 'wet_weather_score', 'boardings', 'alightings', 'is_urban']   # array to be used with scatterplot function below

# Creates 2D scatterplots  
def scatter_plot(y_var, x_var, dataframe):
	y_label = y_var
	x_label = x_var 
	sns_scatterplot = sns.scatterplot(y=dataframe[y_var], x= dataframe[x_var], data=dataframe, s=2)
	sns_scatterplot.set_xlabel(x_label)
	sns_scatterplot.set_ylabel(y_label)
	#handles, labels = sns_scatterplot.get_legend_handles_labels()
	#sns_scatterplot.legend()
	fig3 = sns_scatterplot.get_figure()
	fig3.savefig('plots/' + y_label + '_' + x_label + '_scatterplot.png', dpi = 300)
	fig3.clf()

def plot_percentage_hourly(day, result):
	plt.figure() 
	plt.title('OTP for ' + str(day))
	names = list(result.keys() )
	percentages = list(result.values() )
	plt.bar(range(len(result)), percentages, tick_label = names)
	plt.ylabel('Percentages(%)')
	plt.xlabel('hour be' + str(day))
	plt.xticks(rotation=90)
	plt.draw()
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
	plt.draw()
	plt.savefig('plots/OTP_day_' + str(day) + '.png')
	plt.figure().clear()
	plt.close()
	plt.cla()
	plt.clf()
















