
"""
pretrained model from TASK 2.1, train on mnist dataset

The main objective of this file is to train the pretrained model from Task 2.1 on MNIST train dataset and
evaluate it on the MNIST test dataset

"""

# Importing the modules
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



# Data loading and preprocessing 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0




# Customizing the model from Task 2.1 for 10 classes
model = main_model()
base_inputs = model.layers[0].input
base_outputs = model.layers[-2].output
final_output = layers.Dense(10)(base_outputs)
new_model = keras.Model(inputs = base_inputs, outputs = final_output)
print(new_model.summary())





# Training the pretrained model on MNIST dataset
new_model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits= True),
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"] ) 






save_callback = keras.callbacks.ModelCheckpoint(
    "C:/Users/Asus/Desktop/machine learning/MIDAS IIIT-D challenge/Task 2.3/checkpoint_mnist_pretrained2.1/", save_weights_only=True, monitor="train_acc", save_best_only=False,
)





TB_PATH = "C:/Users/Asus/Desktop/machine learning/MIDAS IIIT-D challenge/Task 2.3/mnist_pretrained2.1_callback_dir"
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=TB_PATH, histogram_freq=1
)




new_model.fit(x_train, y_train, batch_size=64, callbacks=[save_callback, tensorboard_callback],epochs=10, verbose=2)




new_model.evaluate(x_test, y_test, batch_size=64, verbose=2)





Y_pred = model.predict_generator(x_test,10000//64+1)
y_pred = np.argmax(Y_pred, axis=1)

print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred))


