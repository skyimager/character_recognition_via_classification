from keras.layers import Input, Dropout, Dense, GlobalAveragePooling2D, concatenate
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import  Model
from keras.optimizers import Nadam


# Modular function for Fire Node
def fire_module(x, squeeze=16, expand=64):

    x = Conv2D(squeeze, kernel_size=(1, 1), padding='valid', activation='relu')(x)   

    left = Conv2D(expand, kernel_size=(1, 1), padding='valid', activation='relu')(x)

    right = Conv2D(expand, kernel_size=(3, 3), padding='same', activation='relu')(x)
    
    x = concatenate([left, right], axis=-1)
    
    return x


def build(size, chs, lr = 0.001, classes = 1):
    """Instantiates the SqueezeNet architecture.
    """
    
    activation = 'sigmoid'
    loss_func = 'binary_crossentropy'
    
    input_layer = Input(shape=(size, size, chs))

    x = Conv2D(64, kernel_size=(3, 3), strides=(2,2), padding='valid', activation='relu') (input_layer)
    x = MaxPooling2D(pool_size=(3, 3),strides=(2,2))(x)

    x = fire_module(x, squeeze=16, expand=32)
    x = fire_module(x, squeeze=16, expand=32)
    maxp_1 = MaxPooling2D(pool_size=(3, 3),strides=(2,2))(x)

    x = fire_module(maxp_1, squeeze=32, expand=64)
    x = fire_module(x, squeeze=32, expand=64)
    maxp_2 = MaxPooling2D(pool_size=(3, 3),strides=(2,2))(x)

    x = fire_module(maxp_2, squeeze=48, expand=192)
    
    x = Dropout(0.25, name='drop9')(x)

    conv = Conv2D(2, (1, 1), padding='valid', activation='relu')(x)
    gap = GlobalAveragePooling2D()(conv)
    
    predictions = Dense(classes, activation=activation)(gap)
    
    model = Model(inputs=input_layer, outputs=predictions)

    optimizer = Nadam(lr=lr)

    model.compile(optimizer=optimizer, loss=loss_func, metrics=['acc'])
    
    #161,173 parameters
    print(model.summary())    

    return model

