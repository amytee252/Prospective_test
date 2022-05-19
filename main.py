#### Script that contains solutions to Prospective test
import numpy as np
import pandas as pd


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

for day in range(days):
	print(day)
	df_day[day] = pd.DataFrame()
	df_day[day] = grouped_day.get_group(grouped_day)
	print(df_day[day])
	#for trip in range(trip_num):
	#	df_trip[trip] = pd.DataFrame()
	#	print(df[trip])
	





