import numpy as np
import sys
sys.path.append('/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/MSU REU/MSU REU Project/python/')
import ProcessingData as data
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

filename = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/MSU REU/MSU REU Project/BetaScint2DEnergy.csv'
grid, outputs = data.get_data(filename)

relative = data.relative_positions(outputs)



batch_size = 128
num_classes = 100
epochs = 10

x_train, y_train, x_test, y_test = data.cnn_train_test_data(20000,1000, outputs)

x_train = x_train.reshape(20000,72,72,1)
x_test = x_test.reshape(1000,72,72,1)

y_training = relative[:20000]
y_testing = relative[20000:21000]


print('Finished making the training and testing data, here are the stats:')
print('training set input shape: %s' %(str(x_train.shape)))
print('training set output shape: %s' %(str(y_train.shape)))
print('testing set input shape: %s' %(str(x_test.shape)))
print('testing set output shape: %s' %(str(y_test.shape)))

# CNN MODEL
#-----------------------------------------------------------------------------
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation = 'relu', input_shape= (72,72,1)))
model.add(Conv2D(64, (3,3), activation = 'relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.25))
#model.add(Dense(128, activation = 'relu'))
#model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512, input_dim=100, activation='relu'))
model.add(Dense(1, activation='relu'))
model.compile(loss='mse', optimizer='adam', metrics = ['accuracy'])

model.fit(x_train, y_training,
        batch_size= batch_size,
        epochs = epochs,
        verbose = 1, validation_data = (x_test, y_testing))
model.summary()

feed_forward = model.predict(x_testing)

for i in range(len(feed_forward)-900):
    print('Correct: %f \t Predicted: %f' %(y_testing[i], feed_forward[i]))
