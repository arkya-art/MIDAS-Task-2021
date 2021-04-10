# MIDAS IIIT-D 2021 Task 2 

## Problem

* Use this [dataset](https://www.dropbox.com/s/pan6mutc5xj5kj0/trainPart1.zip) to train a CNN. Use no other data source or pretrained networks, and explain your design choices during preprocessing, model building and training. Also, cite the sources you used to borrow techniques. A test set will be provided later to judge the performance of your classifier. Please save your model checkpoints.
* Next, select only 0-9 training images from the above dataset, and use the pretrained network to train on MNIST dataset. Use the standard MNIST train and test splits (http://yann.lecun.com/exdb/mnist/). How does this pretrained network perform in comparison to a randomly initialized network in terms of convergence time, final accuracy and other possible training quality metrics? Do a thorough analysis. Please save your model checkpoints.
* Finally, take the following dataset (https://www.dropbox.com/s/otc12z2w7f7xm8z/mnistTask3.zip), train on this dataset and provide test accuracy on the MNIST test set, using the same test split from part 2. Train using scratch random initialization and using the pretrained network part 1. Do the same analysis as 2 and report what happens this time. Try and do qualitative analysis of what's different in this dataset. Please save your model checkpoints.

# File Path
 
* .ipynb_checkpoints
#### Checkpoint of experiment log.ipynb 

* _pycache
#### Path files for model.py, datagen.py, visualization.py

* Task 2.1

### visualization.py
In this file, I am plotting the images using matplotlib library.
In the data_visualization function, passing the folder path to the images
within the folder path we have many labelled files and within every file we have 
images inside it. [visualization](https://github.com/arkya-art/MIDAS-Task-2021/blob/master/Task%202.1/visualization.py)

#### model.py
The main objective of this file is to design the main model for the task and would use this model in later stages
by simply invoking the function call. [model](https://github.com/arkya-art/MIDAS-Task-2021/blob/master/Task%202.1/model.py)

#### main.py
The main objective of this file is to load the functions from previously build python files and combine
them to get the dataset, get the model and train the model using the data and evaluate its performance 
on the validation set. [main](https://github.com/arkya-art/MIDAS-Task-2021/blob/master/Task%202.1/main.py)

#### datagen.py
The main objective of this file is to collect the dataset from the folders and split
it into training and validation sets. [datagen](https://github.com/arkya-art/MIDAS-Task-2021/blob/master/Task%202.1/datagen.py)

#### model checkpoint
* Task 2.2
* Task 2.3
* train
* .gitignore
* experiment log



 
 
 


