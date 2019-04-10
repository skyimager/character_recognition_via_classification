"""
Source:
    https://github.com/FlorianMuellerklein/Chars74k_CNN/blob/master/chars74k_cnn.py

"""
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D

def build_model(input_shape=(64,64,1), nclasses=62):

    if nclasses > 1:
        activation = "softmax"
    else:
        activation="sigmoid"


    model = Sequential()

    model.add(Convolution2D(128,3,3, input_shape=input_shape, activation = 'relu'))
    model.add(Convolution2D(128,3,3, activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(256,3,3, activation = 'relu'))
    model.add(Convolution2D(256,3,3, activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(512,3,3, activation = 'relu'))
    model.add(Convolution2D(512,3,3, activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # convert convolutional filters to flat so they can be feed to fully connected layers
    model.add(Flatten())

    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nclasses))
    model.add(Activation(activation))

    return model

#    # setting sgd optimizer parameters
#    sgd = SGD(lr=0.03, decay=1e-4, momentum=0.9, nesterov=True)
#    model.compile(loss='categorical_crossentropy', optimizer=sgd)
