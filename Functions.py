#Script to contain the non-plotting functions used across the three main scripts.
import numpy as np

import pandas as pd
from pandas import Series

import seaborn

import matplotlib.pyplot as plt

def time_convert(dataframe, column_name):
	print(column_name)
	dataframe[column_name] = pd.to_datetime(dataframe[column_name], format =  '%H:%M:%S')
	dataframe[column_name] = dataframe[column_name].dt.hour * 3600 + dataframe[column_name].dt.minute * 60 + 			 	 	dataframe[column_name].dt.second


def df_to_dataset(dataframe,  batch_size=1):  #Function to convert the dataframe to a tensorflow dataframe that can be ML'd on. Temporarily remove the target column
	dataframe = dataframe.copy()
	new_df = dataframe.pop("dwell_time")
	new_df = dataframe.pop("is_urban")
	new_df = tf.convert_to_tensor(new_df)
	return new_df

