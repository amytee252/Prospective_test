#### Script that contains solutions to Prospective test for task T1Q1
import numpy as np

import pandas as pd
from pandas import Series

import seaborn

import matplotlib.pyplot as plt

from Functions import *
from Plots import *

### T1Q1. Calculate and plot the daily OTP for route 56 during March 2022. i.e. report_data vs. OTP percentage. Note the OTP is expected to vary day by day and be systematically different on weekends.

df_TO = pd.read_csv("./prospective-ds-home-challenge/datasets/timing_observations.csv")
print(df_TO)
print(df_TO.info())

#I suspect this data cleaning is pointless, but good to do it anyways for reassurance
# March has 31 days
# Want to select only rows of data that occur in March
start_date = '2022-03-01'
end_date = '2022-03-31'
mask = (df_TO['report_date'] >= start_date ) & (df_TO['report_date'] <= end_date)  #Create a mask which only keeps data between two dates
df_TO_2 = df_TO.loc[mask] #Apply mask, which creates a new dataframe with selected rows of data.
print(df_TO_2)
df_TO_2.dropna() # Remove any empty rows of date
print(df_TO_2)



# Want to only keep rows of data that contain 'true' in the timing_point column
df_TO_2 = df_TO_2[df_TO_2.timing_point] #This does the trick
print('removing the false')
print(df_TO_2['timing_point'])
print(df_TO_2[~df_TO_2['timing_point']]) #As a check, print to see if there are any 'false' values remaining in dataframe to see if above implemented correctly. There shouldn't be.


days = df_TO_2['report_date'].unique()  #store all unique days
days_to_int = {day: i  for i, day in enumerate(days)}
int_to_days = {i: day  for i, day in enumerate(days)}
print(days_to_int)
print(int_to_days)
df_TO_2 = df_TO_2.replace(days_to_int)
print(df_TO_2)
print(df_TO_2.info())

grouped_day = df_TO_2.groupby(['report_date']) 

#create dictionaries to hold multiple dataframes
df_day = {}
df_trip = {}

days = df_TO_2['report_date'].nunique()


time_convert(df_TO_2, 'scheduled_start_time')
time_convert(df_TO_2, 'scheduled_arrival')
time_convert(df_TO_2, 'scheduled_departure')
time_convert(df_TO_2, 'observed_arrival')
time_convert(df_TO_2, 'observed_departure')

print(df_TO_2)

for day in range(days):
	#print(day)
	df_day = pd.DataFrame()  #Create an empty dataframe
	df_result = pd.DataFrame()
	df_day = grouped_day.get_group(day) #Fill dataframe with only one day's rows of data, e.g day 1 only
	trips = df_day['trip_id'].unique() #Get the unique trip ids
	trips_to_int = {trip: i for i, trip in enumerate(trips)} #convert each unique trip id to a number
	df_day= df_day.replace(trips_to_int) # replace each trip id with a number
	#print(df_day[day])
	grouped_trip = df_day.groupby(['trip_id']) #group rows of data by their trip_id
	#print(grouped_trip)
	trips = df_day['trip_id'].nunique() # Get the number of unique trip ids
	print('trips: ', trips)
	result = {}
	#print(df_day[day])
	for trip in range(trips):
		print('trip ', trip)
		df_trip = pd.DataFrame() #Create an empty dataframe
		df_trip = grouped_trip.get_group(trip) # Fill will information of a single trip_id.
		max_value = df_trip['sequence_number'].max()
		number_of_stops = df_trip.shape[0]
		#print('max sequence number: ', max_value)
		#print('min sequence number: ', min_value)
		if max_value:
			df_trip['difference'] = abs(df_trip['scheduled_arrival'] - df_trip['observed_arrival'])
		else:
			df_trip['difference'] = abs(df_trip['scheduled_departure'] - df_trip['observed_departure'])
		#print(df_trip)
	#Calculate OTP
		df_trip = df_trip.drop(df_trip[df_trip.difference < 300].index)
	# OTP is [(total number of stops considered - total number of late stops)/ total number of stops considered)*100]
	
		OTP = round(( ((number_of_stops - df_trip.shape[0]) / (number_of_stops)) * 100), 0)
		print(OTP)
		result[trip] = OTP
	print(result)
	for k,v in int_to_days.items():
		if k == day:
			day = v
	plot_percentage_day(day, result)
	


	
	
		
		
		
		
		
	





