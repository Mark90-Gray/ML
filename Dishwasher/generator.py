import tensorflow as tf
import numpy as np

'''Link riferimento: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly'''

class DataGenerator(tf.keras.utils.Sequence):

  'Generates data for Keras'
  
  def __init__(self, main, label, window_len, batch_len, shuffle=True):
    'Initialization'
    self.main = main
    self.label = label
    self.window_len = window_len
    self.batch_len = batch_len
    self.indices = np.arange(len(self.main) - self.window_len - 1)
    self.shuffle = shuffle
    
    

  def __len__(self):
    'Denotes the number of batches per epoch'
    return int(len(self.indices) / self.batch_len) + 1
  
  
  def __getitem__(self, index):
    'Generate one batch of data'
    main_batch = []
    labels_batch = []
    if index == self.__len__() - 1:
      indexes = self.indices[index * self.batch_len:] 
    else:
      indexes = self.indices[index * self.batch_len: (index + 1) * self.batch_len] 
      
    for i in indexes:
      main_sample = self.main[i:i+self.window_len]
      label_sample = self.label[i+int(self.window_len/2)+1]
      main_batch.append(main_sample)
      labels_batch.append(label_sample)

    main_batch = np.array(main_batch)
    main_batch = np.reshape(main_batch, (-1, main_batch.shape[1], 1))
    labels_batch = np.array(labels_batch)
      
    return main_batch, labels_batch

  def on_epoch_end(self):
    'Updates indexes after each epoch'
    
    if self.shuffle == True:
      np.random.shuffle(self.indices)
      
'''
def __data_generation(self, list_IDs_temp):
  'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
  # Initialization
  X = np.empty((self.batch_size, *self.dim, self.n_channels))
  y = np.empty((self.batch_size), dtype=int)

  # Generate data
  for i, ID in enumerate(list_IDs_temp):
      # Store sample
      X[i,] = np.load('data/' + ID + '.npy')

      # Store class
      y[i] = self.labels[ID]

  return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
'''