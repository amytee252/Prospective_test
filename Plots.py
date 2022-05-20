#Script that contains plotting functions across the three main scripts


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
