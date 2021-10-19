import numpy as np
from prettytable import PrettyTable
from math import ceil

''' tutto ciò che serve al preprocessing'''

def load_data(df_data):
  dataset = df_data['power']
  dataset.index = df_data['timestamp']
  return dataset

def split_data(data):
  timestamp='2019-03-01 00:00:00' #circa 80% del dataset
  train=data[:timestamp].values
  val=data[timestamp:].values
  return train, val
  
def standardizzazione(data, mean, std):
  data = data - mean
  data = data/std
  return data

def normalizzazione(data, min_data, max_data):
  data = data - min_data
  diff = max_data - min_data
  data = data/diff
  return data

def reverse_norm(data, min_data, max_data):
  diff = max_data - min_data
  data = data * diff
  data = data + min_data
  return data
  
def reverse_strd(mean, std, data):
  data = data * std
  data = data + mean
  data[data < 0.0] = 0.0 
  return data
  
def padding(a, wind):
  sizeWind= int(wind/2)+1
  b= np.zeros(sizeWind)
  res= np.concatenate((a, b), axis=None)
  res2= np.concatenate((b, res), axis=None)
  result=np.reshape(res2,(res2.shape[0],1))
  return result



def stat(nome1, nome2, array1, array2):
  mm1=array1.mean()
  mm2=array2.mean()
  std1=array1.std()
  std2=array2.std()
  t = PrettyTable(['', nome1, nome2])
  t.add_row(['mean',mm1, mm2])
  t.add_row(['std', std1, std2])
  print(t)
  
def statDish(nome1, nome2, array1, array2):
  mm1=array1.min()
  mm2=array2.min()
  std1=array1.max()
  std2=array2.max()
  t = PrettyTable(['', nome1, nome2])
  t.add_row(['min',mm1, mm2])
  t.add_row(['max', std1, std2])
  print(t)