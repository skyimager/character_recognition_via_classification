#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from keras.layers import Conv2D, Dropout, MaxPooling2D, Input, Flatten, Dense
from keras.layers import BatchNormalization, Activation
from keras.models import Model

def double_conv(x, filters):
    
    conv = Conv2D(filters, (3, 3), padding='same')(x)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
        
    conv = Conv2D(filters, (3, 3), padding='same')(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Dropout(0.2)(conv)
    
    conv = MaxPooling2D((2, 2))(conv)
    
    return conv

def build(input_shape=(64, 64, 1), nb_classes = 62):
    
    filters=64
    inp = Input(input_shape)
    
    x= double_conv(inp,filters)
    x= double_conv(x,filters*4)
    x= double_conv(x,filters*8)
    
    flat = Flatten()(x)

    x = Dense(256, kernel_initializer="he_normal")(flat)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    out = Dense(nb_classes, activation='softmax')(x)
    
    model = Model(inputs=inp, outputs=out)
    
    return model
