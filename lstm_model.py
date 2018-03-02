from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, Input
from keras.layers import LSTM
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf

def model_l():
	input_word = Input(batch_shape=(None, 100, 7), dtype='float32', name='input')
	hidden_units1 = LSTM(units=250, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True,  
					kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', 
					unit_forget_bias=True, dropout=0.1, recurrent_dropout=0.1, implementation=1, return_sequences=True, 
					name='lstm1')(input_word)
	
	
	hidden_units2 = LSTM(units=250, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True,  
					kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', 
					unit_forget_bias=True, dropout=0.1, recurrent_dropout=0.1, implementation=1, return_sequences=True, 
					name='lstm2')(hidden_units1)
	
	
	predictions = TimeDistributed(Dense(3, activation='sigmoid'), name='predictions')(hidden_units2)
	
	model = Model(input=input_word, output=predictions)
	
	optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.1, amsgrad=False)
	model.compile(loss='mae', optimizer=optimizer, metrics=['mae', 'acc'])
	
	#model.fit_generator(train_data_generator.generate(), 6, 50, validation_data=valid_data_generator.generate(), validation_steps=1)
	#model.load_weights('lstm_reg.model', by_name=True)
	#model.save('lstm_reg.model')

	return model