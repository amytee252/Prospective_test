# Background

## Task 1

Task T1Q1 can be found in script `T1Q1.py` and is run with `python T1Q1.py`.
Similarly, task T1Q2 is found in script `T1Q2.py` and is run with `python T1Q2.py`.

The first dataset, `timing_observations.csv` contains scheduled and observed stop arrivals and departures for Route 56. Task T1Q1 asks to calculate and plot the daily OTP during March 2022. 

To complete this task, a number of data cleaning operations are performed, although I suspect are redundant:
1) Remove all dates outside of March 2022
2) Remove all stops that should be be considered, and check to make sure that the dataframe is just left with the relevant stops
3) Check there is no missing data

I also converted all the timing data to seconds, taking a full day to contain 86400 seconds.

For each day (there are 31 in total) I work out the OTP for trip\_id. This is because Route 56 happens many times a day, and so the OTP is worked out for each trip\_id. Note that as the trip\_id is a long string, each unique trip\_id is converted to a number, where the first trip of the day is identifed by 0, and increments by 1 for each new trip\_id. As such, there is a plot for each day, with each plot containing the OTP for each trip\_id that occured that day.


Task T1Q2 asks to only use data for weekdays, calculate and plot the hourly OTP for Route 56, using scheduled_start_time to bin the data. This question was a little ambigious, as one could create plots for each date, presenting the OTP in terms of binned scheduled_start_date just for that day. What I choose to do, was actually to look more at the averages. So after removing the weekends from the data, I am left with five week days. Taking, for example, Monday, I took all the data associated with all the Mondays, and calculated the OTP for each hour that there was a bus scheduled to operate. As such, an hour will represent several Mondays. Hence, we can see how often a bus was on time between the hours of say 5am-6am every Monday. This will be represented by the x-axis label '5'. This is repeated for all the days of the week.

When doing this task, it was assumed that the data was already ordered. I also rounded numbers where appropriate.

## Task 2

This task is run with `python T2.py`

The aim of the task was to building a machine learning (ML) model that predicted the `dwell_time` of a bus, with the ultimte aim of trying to deduce whether or not once could identify whether a bus stopped at an urban bus stop or not (although I never got this far!). I implemented a regression model using a forward feed neural network. I found that keeping the network very simple with one input later, one hidden layer, and a single output layer was enough. The number of nodes in the input layer and hidden layer corresponds to the number of features. There is a single output node, as we are trying to predict a single time here, as the target we are trying to predict is the dwell_time. 

The model trains on the training data of which there is a 80/20 train/test split, trying to learn the patterns in the data, to then predict the dwell\_time on the test dataset. 

Due to time constraints there has not been much parameter optimisation, although generally for regression problems there is a somewhat standard set of parameters. Hence, the input and hidden layer, both use the `ReLu` activation function, the `Adam` optimiser is used, and the learning rate is set to 0.0001. These parameters yield good results, as a plot is made of the true test dwell\_time values vs. the predicted dwell\_time values (using the test set), and it is very close, with values lying on the diagonal, the residuals are plotted too (but they are tiiiiinnnnnnyyyyy).

As we are doing regression, there is no activation function in the output layer, as we want a linear transform (i.e no multiplication).


# Requirements

This requires a number of packages to be installed and run in a virtual enviornment. Please follow the instructions here:

```
python3 -m venv myenvNN
source myenvNN/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```


