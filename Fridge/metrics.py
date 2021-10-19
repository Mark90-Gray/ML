import tensorflow as tf
from keras import backend as K
import tensorflow.math as tm

'''
Link di riferimento: https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/metrics.py
'''

class F1_score(tf.keras.metrics.Metric):

  def __init__(self, scaling=None, name='f1_score', Min=0.0, Max=1.0, Mean=0.0, Std=1.0, **kwargs):
   
    super(F1_score, self).__init__(name=name, **kwargs)
    
    self.sum_pred_true = self.add_weight(name='sum_pred_true', initializer='zeros')
    self.sum_pred = self.add_weight(name='sum_pred', initializer='zeros')
    self.sum_true = self.add_weight(name='sum_true', initializer='zeros')
    self.Min = Min
    self.Max = Max
    self.Mean = Mean
    self.Std = Std
    self.scaling = scaling
    
    
  def update_state(self,x, y, sample_weight=None):
    '''
    Standardizzazione e Normalizzazione inversa se necessario
    '''
    if self.scaling == 'norm':
      y_original = y * (self.Max - self.Min)
      y_original += self.Min
      x_original =x * (self.Max - self.Min)
      x_original += self.Min
    elif self.scaling == 'strd':
      y_original = y * (self.Std)
      y_original += self.Mean
      x_original =x * (self.Std)
      x_original += self.Mean
    else:
      y_original = y
      x_original = x
      
    '''
    Azzera gli eventuali valori negativi
    '''
    neg_indexes = tf.less(y_original, tf.constant(0.0))
    y_positive = tf.where(neg_indexes, tf.zeros_like(y_original), y_original)
    
    '''
    Somma i nuovi dati 
    '''
    self.sum_pred.assign_add(tm.reduce_sum(y_positive))
    self.sum_true.assign_add(tm.reduce_sum(x_original))
    self.sum_pred_true.assign_add(tm.reduce_sum(tm.minimum(y_positive,x_original)))
    
  def result(self):
    """
    Calcola f1 alla fine di ogni epoca
    """
    precision = tm.divide(self.sum_pred_true, self.sum_pred + K.epsilon())
    recall = tm.divide(self.sum_pred_true, self.sum_true + K.epsilon())
    return tm.divide(2 * tm.multiply(precision, recall), tm.add(precision, recall))

  def reset_states(self):
    '''
    Azzera tutto
    '''
    self.sum_pred_true.assign(0.0)
    self.sum_pred.assign(0.0)
    self.sum_true.assign(0.0)
    

def f1_score(y_true, y_pred):
  '''
  Calcola l'F1 dopo la predict
  '''
  def precision(y_true, y_pred):
    sum = 0
    sum_pred = 0
    #for i in range(0, y_true.shape[0]):
    for i in range(0, len(y_true)):
      sum = sum + min(y_true[i], y_pred[i])
      sum_pred = sum_pred + y_pred[i]
    return (sum / sum_pred)

  def recall(y_true, y_pred):
    sum = 0
    sum_true = 0
    #for i in range(0, y_true.shape[0]):
    for i in range(0, len(y_true)):
      sum = sum + min(y_true[i], y_pred[i])
      sum_true = sum_true + y_true[i]
    return (sum / sum_true)    
  P = precision(y_true, y_pred)
  print('precision: ',P)
  R = recall(y_true, y_pred)
  print('recall: ',R)
  return 2*((P*R)/(P + R))
