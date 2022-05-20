### Script that contains solutions to T1Q2

import numpy as np

import pandas as pd
from pandas import Series

import seaborn

import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay

import array as arr

from Functions import *
from Plots import *


## T1Q2: Only including data for weekdays, calculate and plot the hourly OTP for route 56. Use the schedule_start_time hour to bin the data.
df_TO = pd.read_csv("./prospective-ds-home-challenge/datasets/timing_observations.csv")
print(df_TO)
print(df_TO.info())

# March has 31 days
# Want to select only rows of data that occur in March
start_date = '2022-03-01'
end_date = '2022-03-31'
mask = (df_TO['report_date'] >= start_date ) & (df_TO['report_date'] <= end_date)  #Create a mask which only keeps data between two dates
df_TO_2 = df_TO.loc[mask] #Apply mask, which creates a new dataframe with selected rows of data.
print(df_TO_2)
df_TO_2.dropna() # Remove any empty rows of date
print(df_TO_2)


df_TO_2['report_date'] = pd.to_datetime(df_TO_2.report_date, format = '%Y-%m-%d') 
df_TO_2 = df_TO_2[df_TO_2.report_date.dt.weekday < 5] #Remove all weekends from data


# Want to only keep rows of data that contain 'true' in the timing_point column
df_TO_2 = df_TO_2[df_TO_2.timing_point] #This does the trick
print('removing the false')
print(df_TO_2['timing_point'])
print(df_TO_2[~df_TO_2['timing_point']]) #As a check, print to see if there are any 'false' values remaining in dataframe to see if above implemented correctly. There shouldn't be.

#Use the feature scheduled_start_time to create a new column which shows the hour that the bus was scheduled to start its route at
#Also create a new column which gets the day the bus trip took place
df_TO_2['scheduled_start_time'] = df_TO_2['scheduled_start_time'].astype(str)
df_TO_2['time_bin'] = df_TO_2['scheduled_start_time'].str[:2]
df_TO_2['time_bin'] = df_TO_2['time_bin'].astype(int)
df_TO_2['day_of_week'] = df_TO_2['report_date'].dt.day_name()
df_TO_2['day_of_week'] = df_TO_2['day_of_week'].astype(str)

days = df_TO_2['report_date'].unique()  #store all unique days
days_to_int = {day: i  for i, day in enumerate(days)}
int_to_days = {i: day for i, day in enumerate(days)} #..and vice versa! integers(key) and original subject IDs(values)
print(days_to_int)
print(int_to_days)
df_TO_2 = df_TO_2.replace(days_to_int) #replace the report date with an integer
print(df_TO_2)
print(df_TO_2.info())

grouped_day = df_TO_2.groupby(['report_date']) 

#create dictionaries to hold multiple dataframes
df_day = {}
df_trip = {}

days = df_TO_2['day_of_week'].nunique()

#convert time features to seconds
time_convert(df_TO_2, 'scheduled_start_time')
time_convert(df_TO_2, 'scheduled_arrival')
time_convert(df_TO_2, 'scheduled_departure')
time_convert(df_TO_2, 'observed_arrival')
time_convert(df_TO_2, 'observed_departure')

print(df_TO_2)
print(df_TO_2.info() )

#create dictionaries to hold multiple dataframes
df_day = {}
df_hour = {}


for weekday in range(days):
	print('weekday ', weekday)
	print(weekday)
	df_day = pd.DataFrame()  #Create an empty dataframe
	df_day = grouped_day.get_group(weekday) #Fill dataframe with only one one of 5 days of data
	hours = df_day['time_bin'].unique() # Get the number of unique trip ids
	hours_to_int = {hour: i for i, hour in enumerate(hours)} #convert each hour to a number
	int_to_hours = {i: hour for i, hour in enumerate(hours)} #convert each int to an hour
	print(int_to_hours)

	df_day= df_day.replace(hours_to_int) # replace each hour with a number
	result = {}
	print(df_day)
	grouped_hour = df_day.groupby(['time_bin']) #group rows of data by their time bin
	hours = df_day['time_bin'].nunique() # Get the number of unique time bins
	for hour in range(hours):
		df_hour = pd.DataFrame()
		df_hour = grouped_hour.get_group(hour) # Fill will information of a single hour
		print('hour ', hour)
		max_value = df_hour['time_bin'].max()
		number_of_stops = df_hour.shape[0]
		#print('max sequence number: ', max_value)
		#print('min sequence number: ', min_value)
		if max_value:
			df_hour['difference'] = abs(df_hour['scheduled_arrival'] - df_hour['observed_arrival'])
		else:
			df_hour['difference'] = abs(df_hour['scheduled_departure'] - df_hour['observed_departure'])
		#print(df_trip)
	#Calculate OTP
		df_hour = df_hour.drop(df_hour[df_hour.difference < 300].index)  #drop all on time 
	# OTP is [(total number of stops considered - total number of late stops)/ total number of stops considered)*100]
	
		OTP = round(( ((number_of_stops - df_hour.shape[0]) / (number_of_stops)) * 100),0)
		print(OTP)
		result[hour] = OTP
	print(result)
	plot_percentage_hourly(df_day['day_of_week'].iloc[0], result)
