# -*- coding: utf-8 -*-
"""
@author: arkya
Main python file for Task 2.1

********************************************  Objective  ***************************************************************************
The main objective of this file is to load the functions from previously build python files and combine
them to get the dataset, get the model and train the model using the data and evaluate its performance 
on the validation set.


********************************************  Methodology  *************************************************************************
In this file I am first declaring all the paths, for visualizing the training images with folder_path and similarly
the TRAIN_PATH is also mentioned. After this I am calling the functions one by one in steps

1) data_visualization(folder_path) - For visualizing the training images
2) data_gen(TRAIN_PATH) - For generating the training and validation data
3) main_model() - calls the architecture of our model
4) Callback - For saving the checkpoints of the model
5) model.compile() - compiling the model using loss function, optimiser and required evaluation metric
6) model.fit() - Fitting the model on training data
7) model.evaluate() - Evaluating the data on validation set
"""


# Import the modules
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



# image dimensions and batch size
img_height = 28
img_width = 28
batch_size = 64 
# Training Data Visulaization
folder_path = r'C:/Users/Asus/Desktop/machine learning/MIDAS IIIT-D challenge/train/Sample002'
TRAIN_PATH = r'C:/Users/Asus/Desktop/machine learning/MIDAS IIIT-D challenge/train/'




# Data Visualization
for i in range(5):
  data_visualization(folder_path)



  
# Data collection
train_generator, Val_generator = data_gen(TRAIN_PATH)
print(train_generator.class_indices)
print(Val_generator.class_indices)




# Model preperation
model = main_model()
print(model.summary())



# save_callback For saving the checkpoints of the model
save_callback = keras.callbacks.ModelCheckpoint(
    "C:/Users/Asus/Desktop/machine learning/MIDAS IIIT-D challenge/Task 2.1/checkpoint-modified/", save_weights_only=False, monitor="train_acc", save_best_only=True,
)




# Model compilation and fitting
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



model.load_weights("C:/Users/Asus/Desktop/machine learning/MIDAS IIIT-D challenge/Task 2.1/checkpoint-modified/")

loss, acc = model.evaluate(Val_generator, verbose=2)
print("Trained model, accuracy: {:5.2f}%".format(100 * acc))