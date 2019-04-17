#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For now, keras standard implementation is used. 
The script should have the complete architecture for easy understanding. It will 
be implemented soon.

"""
from keras.applications.densenet import DenseNet201
from keras.applications.xception import preprocess_input


def build(input_shape=(128, 128, 3)):
    
    base_model = DenseNet201(input_shape = input_shape, weights='imagenet', 
                             pooling="max", include_top=False)
    
    return base_model

def pre_process(image):
    return preprocess_input(image)

if __name__ == "__main__":
    
    model = build(input_shape=(128, 128, 3)) #min size=32
    model.summary()
    print(len(model.layers)) #no of layers = 708, output.shape = (None, 1920), params=18M+
