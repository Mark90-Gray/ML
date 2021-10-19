import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.layers import  Activation, Conv1D, Dense, Dropout, Flatten
from tensorflow.keras import Model
from tensorflow.keras import optimizers as opt

'''modello fridge'''
def fridge_model(window_len, validation=None):
  model = Sequential()
  model.add(Conv1D(30,10,activation="relu",input_shape=(window_len,1),strides=1))
  model.add(Conv1D(30, 8, activation='relu', strides=1))
  model.add(Conv1D(40, 6, activation='relu', strides=1))
  model.add(Conv1D(50, 5, activation='relu', strides=1))
  model.add(Dropout(.2))
  model.add(Flatten())
  model.add(Dense(1024, activation='relu'))
  model.add(Dropout(.2))
  model.add(Dense(1))
  
  if validation is not None:
    model.compile(optimizer=opt.Adam(lr= 0.0001), loss='mae', metrics=[validation])
  else:
    model.compile(optimizer=opt.Adam(lr= 0.0001), loss='mae')
  
  return model
'''modello dishwasher'''
def dishwasher_model(window_len, validation=None):
  model = Sequential()
  model.add(Conv1D(30,10,activation="relu",input_shape=(window_len,1),strides=1))
  model.add(Conv1D(40, 8, activation='relu', strides=1))
  model.add(Flatten())
  model.add(Dense(512, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))
  
  # default Adama learning rate 0.001
  if validation is not None:
    model.compile(optimizer=opt.Adam(lr= 0.0001), loss='binary_crossentropy', metrics=[validation])
  else:
    model.compile(optimizer=opt.Adam(lr= 0.0001), loss='binary_crossentropy')
  
  return model
  