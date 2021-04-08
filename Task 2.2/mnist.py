# -*- coding: utf-8 -*-
"""
performance on MNIST dataset

@author: arkya
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


# Data loading and preprocessing 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

def my_model():
    inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(64, 3)(inputs)
    x = layers.Conv2D(64, 3)(x)
    x = layers.MaxPooling2D()(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(128, 3)(x)
    x = layers.Conv2D(128, 3)(x)
    x = layers.MaxPooling2D()(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(256, 3)(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(512, activation = 'relu')(x)
    outputs = layers.Dense(10)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

model3 =  my_model()
print(model3.summary())

model3.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits= True),
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"]
) 

save_callback = keras.callbacks.ModelCheckpoint(
    "C:/Users/Asus/Desktop/machine learning/MIDAS IIIT-D challenge/Task 2.2/checkpoint_mnist/", save_weights_only=True, monitor="train_acc", save_best_only=False,
)

TB_PATH = "C:/Users/Asus/Desktop/machine learning/MIDAS IIIT-D challenge/Task 2.2/mnist_callback_dir"
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=TB_PATH, histogram_freq=1
)

model3.fit(x_train, y_train, batch_size=64, callbacks=[save_callback, tensorboard_callback],epochs=10, verbose=2)

model3.evaluate(x_test, y_test, batch_size=64, verbose=2)

Y_pred = model3.predict_generator(x_test,10000//64+1)
y_pred = np.argmax(Y_pred, axis=1)

print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred))