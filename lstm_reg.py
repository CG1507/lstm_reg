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
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, Input
from keras.layers import LSTM
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf

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

class KerasBatchGenerator(object):
	def __init__(self, data, batch_size, time_step):
		self.ci = 0
		self.data = data
		self.time_step = time_step
		self.batch_size = batch_size

	def generate(self):
		while True:
			x = np.zeros((self.batch_size, self.time_step, 7))
			y = np.zeros((self.batch_size, self.time_step, 3))
			
			for i in range(self.batch_size):
				t = []
				t_1 = []
				for j in range(self.time_step):
					if self.ci  + self.time_step >= len(self.data):
					# reset the index back to the start of the data set
						self.ci = 0
					t.append(self.data[self.ci])
					t_1.append(self.data[self.ci + 1][4:])
					self.ci += 1

				x[i, : , :] = t
				y[i, : , :] = t_1

			yield x, y

def main():
	data_cleaning()
	
	# load dataset
	dataset = read_csv('./dataset.csv', header=0, index_col=0)
	values = dataset.values
	
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaled = scaler.fit_transform(values)

	train_data_generator = KerasBatchGenerator(scaled[:30000, :], 10, 100)
	valid_data_generator = KerasBatchGenerator(scaled[30000:35000, :], 10, 100)
	
	model = lstm_model.model_l()
	optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.1, amsgrad=False)
	model.compile(loss='mae', optimizer=optimizer, metrics=['mae', 'acc'])
	tensorboard_callback = TensorBoard(log_dir='./tensorboard', histogram_freq=0, batch_size=200, write_graph=True, write_grads=True, 
										write_images=False, embeddings_freq=1, embeddings_layer_names=['lstm1'], 
										embeddings_metadata=None)
	model.fit_generator(train_data_generator.generate(), 30, 50, validation_data=valid_data_generator.generate(), validation_steps=5,
						callbacks=[tensorboard_callback])
	#model.load_weights('lstm_reg.model', by_name=True)
	model.save('final_lstm_reg.model')
	print('MODEL SAVED!')


if __name__ == "__main__":
	main()