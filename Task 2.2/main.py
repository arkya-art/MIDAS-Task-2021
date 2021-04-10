# -*- coding: utf-8 -*-
"""
@author: arkya
Training the model on 0-9 datset

********************************************  Objective  ***************************************************************************
The main objective of this file is to load the model architecture from previous model and train the model 
on provided dataset for Task 2.2

********************************************  Methodology  *************************************************************************
In this file I am first declaring all the paths, for visualizing the new training images with folder_path and similarly
the new TRAIN_PATH is also mentioned. After this I am calling the functions one by one in steps

1) data_visualization(folder_path) - For visualizing the new training images
2) data_gen(TRAIN_PATH) - For generating the new training and validation data
3) main_model() - calls the architecture of our model and then customizing it according to the 10 output classes
4) Save Callback - For saving the checkpoints of the new model
5) Tensorboard Callback - For visulizing the model in TensorBoard
6) model.compile() - compiling the model using loss function, optimiser and required evaluation metric
7) model.fit() - Fitting the model on training data
8) model.evaluate() - Evaluating the data on validation set

"""
# Importing the Modules
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



# image dimensions and batch size
img_height = 28
img_width = 28
batch_size = 64 
# Training Data Visulaization
folder_path = r'C:/Users/Asus/Desktop/machine learning/MIDAS IIIT-D challenge/Task 2.2/train/Sample010'
TRAIN_PATH = r'C:/Users/Asus/Desktop/machine learning/MIDAS IIIT-D challenge/Task 2.2/train/'



# Data Visualization
for i in range(5):
  data_visualization(folder_path)
 
    
  
 
# Data collection
train_generator, Val_generator = data_gen(TRAIN_PATH)
print(train_generator.class_indices)
print(Val_generator.class_indices)



# Model preperation
model = main_model()
base_inputs = model.layers[0].input
base_outputs = model.layers[-2].output
final_output = layers.Dense(10)(base_outputs)
model = keras.Model(inputs = base_inputs, outputs = final_output)



# saving checkpoints
save_callback = keras.callbacks.ModelCheckpoint(
    "C:/Users/Asus/Desktop/machine learning/MIDAS IIIT-D challenge/Task 2.2/checkpoint/", save_weights_only=True, monitor="train_acc", save_best_only=False,
)



# Model compilation and fitting
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits= True),
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"]
) 



# TensorBoard callback
TB_PATH = "C:/Users/Asus/Desktop/machine learning/MIDAS IIIT-D challenge/Task 2.2/tb_callback_dir"
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=TB_PATH, histogram_freq=1,
)



# Fitting the model on traing data set
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