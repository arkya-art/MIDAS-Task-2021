# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 00:46:51 2021

@author: arkya
"""


# Import statements

import pandas as pd
import numpy as np
import os
import random
import tensorflow as tf
import cv2
from tensorflow import keras
from keras.layers import *
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
folder_path = r'C:/Users/Asus/Desktop/machine learning/MIDAS IIIT-D challenge/train/Sample002'
TRAIN_PATH = r'C:/Users/Asus/Desktop/machine learning/MIDAS IIIT-D challenge/train/'

for i in range(5):
  data_visualization(folder_path)
  
# Data collection
train_generator, Val_generator = data_gen(TRAIN_PATH)

print(train_generator.class_indices)
print(Val_generator.class_indices)

# Model preperation
model = main_model()
print(model.summary())

save_callback = keras.callbacks.ModelCheckpoint(
    "C:/Users/Asus/Desktop/machine learning/MIDAS IIIT-D challenge/Task 2.1/checkpoint/", save_weights_only=True, monitor="train_acc", save_best_only=False,
)

# Model compilation andfitting
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits= True),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"]
) 

model.fit(
    train_generator,
    epochs=30,
    steps_per_epoch=34,
    callbacks=[save_callback],
    verbose=2,
    validation_data = Val_generator,
    validation_steps = 3
)


#model.save('C:/Users/Asus/Desktop/machine learning/MIDAS IIIT-D challenge/Task 2.1/model_version1')
#SAVE_PATH = 'C:/Users/Asus/Desktop/machine learning/MIDAS IIIT-D challenge/Task 2.1/model-V1'

#tf.keras.models.save_model(model, SAVE_PATH, include_optimizer=True)

model.load_weights("C:/Users/Asus/Desktop/machine learning/MIDAS IIIT-D challenge/Task 2.1/checkpoint/")

loss, acc = model.evaluate(Val_generator, verbose=2)
print("Trained model, accuracy: {:5.2f}%".format(100 * acc))