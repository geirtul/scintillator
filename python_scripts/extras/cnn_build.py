# CONVOLUTIONAL NEURAL NETWORK FOR SEAN LIDDICK ONE-ELECTRON DATA
#------------------------------------------------------------------------------
# PURPOSE
#-----------------------------------------------------------------------------
# To use a convolutional neural network to distinguish between one and two
# electron events. In addition, test to see if the model can learn the cell
# for which the electron started in.
#-----------------------------------------------------------------------------


def cnn_initialize(train, test, epochs, outputs):
    import keras
    import sys
    sys.path.append('/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/MSU REU/MSU REU Project/python/')
    import ProcessingData as data

    batch_size = 128
    num_classes = 100
    epochs = epochs

    x_train, y_train, x_test, y_test = data.cnn_train_test_data(train,test, outputs)

    x_train = x_train.reshape(train,72,72,1)
    x_test = x_test.reshape(test,72,72,1)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print('Finished making the training and testing data, here are the stats:')
    print('training set input shape: %s' %(str(x_train.shape)))
    print('training set output shape: %s' %(str(y_train.shape)))
    print('testing set input shape: %s' %(str(x_test.shape)))
    print('testing set output shape: %s' %(str(y_test.shape)))

    return x_train, y_train, x_test, y_test, epochs, batch_size


def cnn_model(x_train, y_train, x_test, y_test, epochs, batch_size):
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten
    from keras.layers import Conv2D, MaxPooling2D
    # CNN MODEL
    #-----------------------------------------------------------------------------
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation = 'relu', input_shape= (72,72,1)))
    model.add(Conv2D(64, (3,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation = 'softmax'))
    model.compile(loss= keras.losses.categorical_crossentropy,
                optimizer = keras.optimizers.Adadelta(),
                metrics = ['accuracy'])

    history = model.fit(x_train, y_train,
            batch_size= batch_size,
            epochs = epochs,
            verbose = 1, validation_data = (x_test, y_test))
    #-----------------------------------------------------------------------------
    score = model.evaluate(x_test, y_test, verbose = 0)
    print('Test loss %.2f' %(score[0]))
    print('Test accuracy %.5f' %(score[1]))

    return model, history, x_test, y_test
