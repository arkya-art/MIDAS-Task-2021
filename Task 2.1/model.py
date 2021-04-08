# -*- coding: utf-8 -*-
"""
Model for Character Recognition

"""

def main_model():
    
    
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.layers import Dense
    #from tensorflow.keras import *
    from tensorflow.keras.models import Sequential, Model

    inputs = keras.Input(shape=(28,28,1))

    x = layers.Conv2D(filters = 64, kernel_size = (3,3), activation="relu")(inputs)
    x = layers.Conv2D(filters = 64, kernel_size = (3,3), activation="relu")(x)
    x = layers.MaxPool2D(pool_size=(2,2))(x)
    x = layers.BatchNormalization()(x)
      
    
    x = layers.Conv2D(filters = 128, kernel_size = (3,3), activation="relu")(x)
    x = layers.Conv2D(filters = 128, kernel_size = (3,3), activation="relu")(x)
    x = layers.MaxPool2D(pool_size=(2,2))(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(filters = 256, kernel_size = (3,3), activation="relu")(x)   
    x = layers.MaxPool2D(pool_size=(2,2))(x) 
    x = layers.Flatten()(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(512, activation = 'relu')(x)
    
    outputs = layers.Dense(62)(x)
    
    model = keras.Model(inputs = inputs, outputs = outputs)
    
    return model
