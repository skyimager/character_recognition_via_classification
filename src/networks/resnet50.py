#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For now, keras standard implementation is used. 
The script should have the complete architecture for easy understanding. It will 
be implemented soon.

"""

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input

def build(input_shape=(227, 227, 3)):
    
    base_model = ResNet50(input_shape = input_shape, weights='imagenet',
                          pooling="max", include_top=False)
    
    return base_model

def pre_process(image):
    return preprocess_input(image)

if __name__ == "__main__":
    
    model = build(input_shape=(227, 227, 3))
    model.summary()
    print(len(model.layers)) #no of layers = 175, output.shape = (None,2048), params=23M+
