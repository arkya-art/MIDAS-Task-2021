# -*- coding: utf-8 -*-
"""
Data-Generator for training and validation

@author: arkya
"""

def data_gen(TRAIN_PATH):
    
    from tensorflow import keras
    from keras.preprocessing import image
    
    img_height = 28
    img_width = 28
    batch_size = 32
    
    train_datagen = image.ImageDataGenerator(
    rescale = 1.0 / 255,
    rotation_range=5,
    zoom_range = 0.1, 
    width_shift_range=0.1, 
    height_shift_range=0.1,
    validation_split=0.1
    )

    train_generator = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="sparse",
        shuffle=True,
        subset="training",
        seed=123,
    ) 
      
    Val_generator = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="sparse",
        shuffle=True,
        subset="validation",
        seed=123,
    )
    
    return train_generator, Val_generator
    
    