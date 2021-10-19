import matplotlib.pyplot as plt
import numpy as np

'''Grafici loss e f1, confronto train e validation'''

def plot_fit(fit_data):
  history = fit_data.history
  plt.title('Loss')
  plt.plot(np.arange(1, len(fit_data.epoch) + 1), history['loss'], marker='o')
  plt.plot(np.arange(1, len(fit_data.epoch) + 1), history['val_loss'], marker='o')
  plt.legend(['train', 'val'])
  plt.show()
  plt.title('F1')
  plt.plot(np.arange(1, len(fit_data.epoch) + 1), history['f1_score'], marker='o')
  plt.plot(np.arange(1, len(fit_data.epoch) + 1), history['val_f1_score'], marker='o')
  plt.legend(['train', 'val'])
  plt.show()