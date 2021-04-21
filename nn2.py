from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy


def defineModel():
	model = Sequential()
	model.add(Dense(100, input_dim=192, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(15, activation='relu'))
	model.add(Dense(4, activation='linear'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	return model


def loadModel():
	try:
		json_file = open('model2.json', 'r')
	except:
		print("Creating JSON file 2 and weights file")
		return defineModel()

	model_json = json_file.read()
	json_file.close()
	model = model_from_json(model_json)
	weightfile_path = "/home/pallenavya/Pictures/2048-master/code/model2.h5"
	model.load_weights(weightfile_path)
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

	return model

def saveModel(model):
	model_json = model.to_json()
	with open("model2.json", "w") as json_file:
	    json_file.write(model_json)
	model.save_weights("model2.h5")

def getQ(model,X):
	return model.predict(numpy.array([X]))[0]

def train(model,X,Y):
	model.fit(numpy.array(X), numpy.array(Y), epochs=1, batch_size=1000,verbose=0)


