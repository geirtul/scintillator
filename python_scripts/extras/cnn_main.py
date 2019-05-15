


import sys
sys.path.append('/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/MSU REU/MSU REU Project/python/')
import ProcessindData as data
sys.path.append('/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/MSU REU/MSU REU Project/python/convolution_neural_network/src/')
from cnn_build import *
import matplotlib.pyplot as plt
import numpy as np


# LOAD DATA
# -----------------------------------------------------------------------------
filename = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/MSU REU/MSU REU Project/BetaScint2DEnergy.csv'
grid, outputs = data.get_data(filename)
# -----------------------------------------------------------------------------


# MAIN CODE
# -----------------------------------------------------------------------------
x_train, y_train, x_test, y_test, epochs, batch_size = cnn_intialize(2000,1000,40, outputs)

history, model, x_test, y_test = cnn_model(x_train, y_train, x_test, y_test, epochs, batchsize)
# -----------------------------------------------------------------------------
model.summary()


# PLOTS
# ----------------------------------------------------------------------------
#fig, ax = plt.subplots(1,2, figsize = (7,5))
#ax[0].set_aspect('auto')
#ax[0].plot(history.history['acc'], label = 'Train')
#ax[0].plot(history.history['val_acc'], label = 'Test')
#ax[0].set_title('Model Accuracy')
#ax[0].set_xlabel('epoch')
#ax[0].set_ylabel('accuracy')
#ax[0].legend(loc = 'best')
#ax[1].set_aspect('auto')
#ax[1].plot(history.history['loss'], label = 'Train')
#ax[1].plot(history.history['val_loss'], label = 'Test')
#ax[1].set_title('Model Loss')
#ax[1].set_xlabel('epoch')
#ax[1].set_ylabel('loss')
#ax[1].legend(loc = 'best')
#fig.subplots_adjust(right = 2)
#plt.show()
