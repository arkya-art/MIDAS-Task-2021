
"""
Model for Character Recognition

****************************************  Objective  **************************************************************
The main objective of this file is to design the main model for the task and would use this model in later stages
by simply invoking the function call

****************************************  Methodology  ************************************************************
In this file I am first importing all the required libraries needed for the model creation within the function.
Since the dimensions of the input images are 28*28*1 so started the model with the input layer with given dimensions.
The model architecture is discussed in detail in readme.md file.

Input -> Conv2D -> Conv2D -> MaxPool2D -> BatchNormalization
Conv2D -> Conv2D -> MaxPool2D -> BatchNormalization
Conv2D -> MaxPool2D -> Flatten -> BatchNormalization -> Dense -> Dense

Input - 
Output - model
"""

def main_model():
    
    
    # importing the modules
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Sequential, Model



    # Model architecture
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
