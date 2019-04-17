#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For now, keras standard implementation is used. 
The script should have the complete architecture for easy understanding. It will 
be implemented soon.

"""

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

def build(input_shape=(139, 139, 3)):
    
    base_model = InceptionV3(input_shape = input_shape, weights='imagenet', 
                             pooling="max", include_top=False)
    
    return base_model

def pre_process(image):
    return preprocess_input(image)

if __name__ == "__main__":
    
    model = build(input_shape=(150, 150, 3)) #min size=75
    model.summary()
    print(len(model.layers)) #no of layers = 312, output.shape = (None, 2048), params=21M+
