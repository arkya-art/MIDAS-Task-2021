# -*- coding: utf-8 -*-
"""
Pretrained model from 0-9 dataset, train on MNIST dataset

********************************************  Objective  ***************************************************************************
The main objective of this file is to load the weights from previously pretrained model on 0-9 dataset images and to train this pre-
trained model on MNIST dataset and then evaluate its performance on MNIST test dataset


********************************************  Methodology  *************************************************************************

 Firstly loaded the pretrained model on 0-9 dataset and also the inbuild Keras MNIST dataset. Trained this pre-trained model on 
 MNIST dataset

- Loading MNIST Dataset
- main_model() - calls the architecture of our model and then customizing it according to the 10 output classes
- Save Callback - For saving the checkpoints of the new model
- Tensorboard Callback - For visulizing the model in TensorBoard
- model.compile() - compiling the model using loss function, optimiser and required evaluation metric
- model.fit() - Fitting the model on training data
- model.evaluate() - Evaluating the data on MNIST test set
"""

# Importing the dataset
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
from tensorflow.keras.datasets import mnist
from sklearn.metrics import classification_report, confusion_matrix
%matplotlib inline


# image dimensions and batch size
img_height = 28
img_width = 28
batch_size = 64 




# Loading pretrained model on 0-9 dataset
model = main_model()
model.load_weights("C:/Users/Asus/Desktop/machine learning/MIDAS IIIT-D challenge/Task 2.2/checkpoint/")




# MNIST Data loading and preprocessing 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0



# Training the pretrained model on MNIST dataset
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits= True),
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"]
) 



# Save callback for saving checkpoints
save_callback = keras.callbacks.ModelCheckpoint(
    "C:/Users/Asus/Desktop/machine learning/MIDAS IIIT-D challenge/Task 2.2/checkpoint_mnist_pretrained/", save_weights_only=True, monitor="train_acc", save_best_only=False,
)



# TensorBoard callback
TB_PATH = "C:/Users/Asus/Desktop/machine learning/MIDAS IIIT-D challenge/Task 2.2/mnist_pretrained_callback_dir"
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=TB_PATH, histogram_freq=1
)



# Model fitting on MNIST dataset
model.fit(x_train, y_train, batch_size=64, callbacks=[save_callback, tensorboard_callback],epochs=10, verbose=2)



# Model evaluation
model.evaluate(x_test, y_test, batch_size=64, verbose=2)


Y_pred = model.predict_generator(x_test,10000//64+1)
y_pred = np.argmax(Y_pred, axis=1)

print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred))