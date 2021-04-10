
"""
Data-Generator for Training and Validation

****************************************  Objective  **************************************************************
In this file, my main objective is collect the dataset from the folders and split
it into training and validation sets.


***************************************  Methodology  *************************************************************
Intially for this task I have assumed image dimensions 28*28*1 (this can be variable)
and have taken a batch size of 32. Then used the keras.preprocessing.image library
to rescale the images and then apply certain data augmentation techniques (mentioned in read.md).
Ues the instance to pass on the training path with target size, batch, color mode (which is grayscale),
class (this contains multi classes), shuffled the dataset and then used seed so that this particular result can 
be regenerated
 
Input - TRAIN_PATH
Output - train_generator, Val_generator
"""

def data_gen(TRAIN_PATH):
    
    # importing the libraries 
    from tensorflow import keras
    from keras.preprocessing import image
    
    # Image dimensions and batch size
    img_height = 28
    img_width = 28
    batch_size = 32
    
    # Instance for rescaling and augmentations
    train_datagen = image.ImageDataGenerator(
    rescale = 1.0 / 255,
    rotation_range=5,
    zoom_range = 0.1, 
    width_shift_range=0.1, 
    height_shift_range=0.1,
    validation_split=0.1
    )

    # Instance for training Data
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
    
    # Instance for Validation Data
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
    
    