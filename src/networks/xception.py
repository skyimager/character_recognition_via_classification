#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input


def build(input_shape=(139, 139, 3)):
    
    base_model = Xception(input_shape = input_shape, weights='imagenet', 
                             pooling="max", include_top=False)
    
    return base_model

def pre_process(image):
    return preprocess_input(image)

if __name__ == "__main__":
    
    model = build(input_shape=(150, 150, 3)) #min size=71
    model.summary()
    print(len(model.layers)) #no of layers = 133, output.shape = (None, 2048), params=20M+
