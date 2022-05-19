#### Script that contains solutions to Prospective test
import numpy as np

import pandas as pd
from pandas import Series



### T1Q1. Calculate and plot the daily OTP for route 56 during March 2022. i.e. report_data vs. OTP percentage. Note the OTP is expected to vary day by day and be systematically different on weekends.

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



# Want to only keep rows of data that contain 'true' in the timing_point column
df_TO_2 = df_TO_2[df_TO_2.timing_point] #This does the trick
print(df_TO_2['timing_point'])
print(df_TO_2[~df_TO_2['timing_point']]) #As a check, print to see if there are any 'false' values remaining in dataframe to see if above implemented correctly. There shouldn't be.


days = df_TO_2['report_date'].unique()  #store all unique days
days_to_int = {day: i  for i, day in enumerate(days)}
print(days_to_int)
df_TO_2 = df_TO_2.replace(days_to_int)
print(df_TO_2)
print(df_TO_2.info())

grouped_day = df_TO_2.groupby(['report_date']) 

#create dictionaries to hold multiple dataframes
df_day = {}
df_trip = {}

days = df_TO_2['report_date'].nunique()

#Convert all time data columns to seconds
#SUPER CLUNKY CONVERT TO FUNCTION!!!!
df_TO_2['scheduled_start_time'] = pd.to_datetime(df_TO_2.scheduled_start_time, format = '%H:%M:%S')
df_TO_2['scheduled_start_time'] = df_TO_2['scheduled_start_time'].dt.hour * 3600 + df_TO_2['scheduled_start_time'].dt.minute * 60 + df_TO_2['scheduled_start_time'].dt.second


df_TO_2['scheduled_arrival'] = pd.to_datetime(df_TO_2.scheduled_arrival, format = '%H:%M:%S')
df_TO_2['scheduled_arrival'] = df_TO_2['scheduled_arrival'].dt.hour * 3600 + df_TO_2['scheduled_arrival'].dt.minute * 60 + df_TO_2['scheduled_arrival'].dt.second

df_TO_2['scheduled_departure'] = pd.to_datetime(df_TO_2.scheduled_departure, format = '%H:%M:%S')
df_TO_2['scheduled_departure'] = df_TO_2['scheduled_departure'].dt.hour * 3600 + df_TO_2['scheduled_departure'].dt.minute * 60 + df_TO_2['scheduled_departure'].dt.second

df_TO_2['observed_arrival'] = pd.to_datetime(df_TO_2.observed_arrival, format = '%H:%M:%S')
df_TO_2['observed_arrival'] = df_TO_2['observed_arrival'].dt.hour * 3600 + df_TO_2['observed_arrival'].dt.minute * 60 + df_TO_2['observed_arrival'].dt.second

df_TO_2['observed_departure'] = pd.to_datetime(df_TO_2.observed_departure, format = '%H:%M:%S')
df_TO_2['observed_departure'] = df_TO_2['observed_departure'].dt.hour * 3600 + df_TO_2['observed_departure'].dt.minute * 60 + df_TO_2['observed_departure'].dt.second

print(df_TO_2)

for day in range(days):
	#print(day)
	df_day[day] = pd.DataFrame()  #Create an empty dataframe
	df_day[day] = grouped_day.get_group(day) #Fill dataframe with only one day's rows of data, e.g day 1 only
	trips = df_day[day]['trip_id'].unique() #Get the unique trip ids
	trips_to_int = {trip: i for i, trip in enumerate(trips)} #convert each unique trip id to a number
	df_day[day] = df_day[day].replace(trips_to_int) # replace each trip id with a number
	#print(df_day[day])
	grouped_trip = df_day[day].groupby(['trip_id']) #group rows of data by their trip_id
	#print(grouped_trip)
	trips = df_day[day]['trip_id'].nunique() # Get the number of unique trip ids
	print('trips: ', trips)
	on_time_counter = 0
	late_time_counter = 0
	#print(df_day[day])
	for trip in range(trips):
		df_trip[trip] = pd.DataFrame() #Create an empty dataframe
		df_trip[trip] = grouped_trip.get_group(trip) # Fill will information of a single trip_id.
	
		
		
		
		
		
	





