# -*- coding: utf-8 -*-
"""
Main Program

@author: arkya
"""

import pandas as pd
import numpy as np
import os
import random
import tensorflow as tf
import cv2
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras import *
from tensorflow.keras.models import Sequential, Model
from keras.preprocessing import image

from model import main_model
from visualization import data_visualization
from datagen import data_gen

from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

img_height = 28
img_width = 28
batch_size = 64 

# Training Data Visulaization
folder_path = r'C:/Users/Asus/Desktop/machine learning/MIDAS IIIT-D challenge/Task 2.2/train/Sample010'
TRAIN_PATH = r'C:/Users/Asus/Desktop/machine learning/MIDAS IIIT-D challenge/Task 2.2/train/'

for i in range(5):
  data_visualization(folder_path)
  
# Data collection
train_generator, Val_generator = data_gen(TRAIN_PATH)

print(train_generator.class_indices)
print(Val_generator.class_indices)

# Model preperation
model = main_model()
#print(model.summary())

"""
base_input = model.layers[0].input
base_output = model.layers[0].output

x = layers.Conv2D(filters = 32, kernel_size = (3,3))(base_output)
x = layers.BatchNormalization()(x)
x = keras.activations.relu(x)
x = layers.MaxPool2D(pool_size=(2,2))(x)

x = layers.Conv2D(filters = 64, kernel_size = (3,3))(x)
x = layers.BatchNormalization()(x)
x = keras.activations.relu(x)
      
x = layers.Flatten()(x)
    
x = layers.Dense(128, activation = 'relu')(x)
x = layers.Dropout(0.5)(x)
final_output = layers.Dense(10)(x)
model2 = keras.Model(inputs = base_input, outputs = final_output)
"""
#print(model2.summary())

# saving checkpoints
save_callback = keras.callbacks.ModelCheckpoint(
    "C:/Users/Asus/Desktop/machine learning/MIDAS IIIT-D challenge/Task 2.2/checkpoint/", save_weights_only=True, monitor="train_acc", save_best_only=False,
)

# Model compilation andfitting
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits= True),
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"]
) 
TB_PATH = "C:/Users/Asus/Desktop/machine learning/MIDAS IIIT-D challenge/Task 2.2/tb_callback_dir"
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=TB_PATH, histogram_freq=1,
)

model.fit(
    train_generator,
    epochs=25,
    steps_per_epoch=5,
    callbacks=[save_callback, tensorboard_callback],
    validation_data = Val_generator,
    validation_steps = 1,
    verbose=2
)

model.load_weights("C:/Users/Asus/Desktop/machine learning/MIDAS IIIT-D challenge/Task 2.2/checkpoint/")

loss, acc = model.evaluate(Val_generator, verbose=2)
print("Trained model, accuracy: {:5.2f}%".format(100 * acc))