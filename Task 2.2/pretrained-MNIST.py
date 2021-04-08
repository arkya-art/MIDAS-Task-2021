# -*- coding: utf-8 -*-
"""
Pretrained model train on MNIST dataset

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
from tensorflow.keras.datasets import mnist
from sklearn.metrics import classification_report, confusion_matrix
%matplotlib inline

img_height = 28
img_width = 28
batch_size = 64 

# Loading pretrained model on 0-9 dataset
model = main_model()
model.load_weights("C:/Users/Asus/Desktop/machine learning/MIDAS IIIT-D challenge/Task 2.2/checkpoint/")

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Training the pretrained model on MNIST dataset
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits= True),
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"]
) 

save_callback = keras.callbacks.ModelCheckpoint(
    "C:/Users/Asus/Desktop/machine learning/MIDAS IIIT-D challenge/Task 2.2/checkpoint_mnist_pretrained/", save_weights_only=True, monitor="train_acc", save_best_only=False,
)

TB_PATH = "C:/Users/Asus/Desktop/machine learning/MIDAS IIIT-D challenge/Task 2.2/mnist_pretrained_callback_dir"
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=TB_PATH, histogram_freq=1
)

model.fit(x_train, y_train, batch_size=64, callbacks=[save_callback, tensorboard_callback],epochs=10, verbose=2)

model.evaluate(x_test, y_test, batch_size=64, verbose=2)


Y_pred = model.predict_generator(x_test,10000//64+1)
y_pred = np.argmax(Y_pred, axis=1)

print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred))