# SETUP FOR CONVOLUTIONAL NEURAL NETWORK FOR SEAN LIDDICK ONE ELECTRON DATA
#-----------------------------------------------------------------------------
# PURPOSE
#------------------------------------------------------------------------------
# In this file, I am creating the dataset of images for the convolutonal neural
# network. I need to create enough data for training and testing purposes.
#
#------------------------------------------------------------------------------
# IMPORT NECESSARY LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/MSU REU/MSU REU Project/python/')
import ProcessingData as pd #personal library of functions for processing data
#------------------------------------------------------------------------------

#IMPORT ELECTRON DATASET
filename = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/MSU REU/MSU REU Project/'
grid, outputs = pd.get_data(filename)

#CONVERT DATA INTO IMAGES (ONLY RUN ONCE TO MAKE DATA)
# Images are saved under the path:
# /Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/MSU REU/MSU REU Project/CNN_DATA
pd.grid2img(grid)
