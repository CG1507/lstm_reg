import file_io
from math import sqrt
import lstm_model
import numpy as np
from numpy import concatenate
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error


def data_cleaning():
	reading_file_pointer = file_io.read_file('./pollution_new.csv')
	writing_file_pointer = file_io.create_file('./dataset.csv')

	index = []
	days = []
	months = []
	years = []
	hours = []
	minutes = []
	humidities = []
	temperatures = []
	ppm = []


	line_no = 1
	for line in reading_file_pointer:
		if line_no == 1:
			line_no += 1
			continue

		words = line.split(',')
		index.append(words[0].strip())
		ppm.append(words[-1].strip())
		temperatures.append(words[-2].strip())
		humidities.append(words[-3].strip())

		date_time = words[1].split(' ')
		dates = date_time[0].split('/')
		
		months.append(dates[0].strip())
		days.append(dates[1].strip())
		years.append(dates[2].strip())

		time = date_time[1].split(':')
		hours.append(time[0].strip())
		minutes.append(time[1].strip())

	file_io.write_line(writing_file_pointer, 'index,year,month,day,hour,hum,temp,ppm\n')

	for i in range(len(index)):
		line = index[i] + ',' + years[i] + ',' + months[i] + ',' + days[i] + ',' + hours[i] + ',' + humidities[i] + ',' +  temperatures[i] + ',' + ppm[i] + '\n'
		file_io.write_line(writing_file_pointer, line)

def main():
	data_cleaning()
	
	# load dataset
	dataset = read_csv('./dataset.csv', header=0, index_col=0)
	values = dataset.values
	
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaled = scaler.fit_transform(values)

	model = lstm_model.model_l()
	model.load_weights('final_lstm_reg.model', by_name=True)

	x = np.zeros((1,100, 7))
	c = 0
	t = []
	for i in range(100):
		t.append(scaled[35000 + i, :7])
	
	x[0,:,:] = t
	yhat = model.predict(x)
	print(yhat)

	inv_yhat = concatenate((x.reshape((100, 7))[:,:4], yhat.reshape((100, 3))), axis=1)
	inv_yhat = scaler.inverse_transform(inv_yhat)
	print(inv_yhat[:,4:])

if __name__ == "__main__":
	main()