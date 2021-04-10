# MIDAS IIIT-D 2021 Task 2 


# Problem

* Use this [dataset](https://www.dropbox.com/s/pan6mutc5xj5kj0/trainPart1.zip) to train a CNN. Use no other data source or pretrained networks, and explain your design choices during preprocessing, model building and training. Also, cite the sources you used to borrow techniques. A test set will be provided later to judge the performance of your classifier. Please save your model checkpoints.

* Next, select only 0-9 training images from the above dataset, and use the pretrained network to train on MNIST dataset. Use the standard MNIST train and test splits (http://yann.lecun.com/exdb/mnist/). How does this pretrained network perform in comparison to a randomly initialized network in terms of convergence time, final accuracy and other possible training quality metrics? Do a thorough analysis. Please save your model checkpoints.

* Finally, take the following dataset (https://www.dropbox.com/s/otc12z2w7f7xm8z/mnistTask3.zip), train on this dataset and provide test accuracy on the MNIST test set, using the same test split from part 2. Train using scratch random initialization and using the pretrained network part 1. Do the same analysis as 2 and report what happens this time. Try and do qualitative analysis of what's different in this dataset. Please save your model checkpoints.




# File Path
* .ipynb_checkpoints
#### Checkpoint of experiment log.ipynb 


* _pycache
#### Path files for model.py, datagen.py, visualization.py



## Task 2.1
#### visualization.py
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
The checkpoints of the model are saved in this folder. [Checkpoint Task 2.1](https://github.com/arkya-art/MIDAS-Task-2021/tree/master/Task%202.1/checkpoint)


## Task 2.2
#### Training images

#### main.py
The main objective of this file is to load the model architecture from previous model and train the model 
on provided dataset for Task 2.2. [main](https://github.com/arkya-art/MIDAS-Task-2021/blob/master/Task%202.2/main.py)

#### checkpoint
Contains the checkpoint of the model in main.py. [checkpoint](https://github.com/arkya-art/MIDAS-Task-2021/tree/master/Task%202.2/checkpoint)

#### mnist.py
The main objective of this file is to obtain the performance of a randomly initialized model on MNIST Dataset [mnist](https://github.com/arkya-art/MIDAS-Task-2021/blob/master/Task%202.2/mnist.py)

#### checkpoint_mnist
Contains the checkpoint of the model in mnist.py [checkpoint_mnist](https://github.com/arkya-art/MIDAS-Task-2021/tree/master/Task%202.2/checkpoint_mnist)

#### pretrained-MNIST.py
The main objective of this file is to load the weights from previously pretrained model on 0-9 dataset images and to train this pre-
trained model on MNIST dataset and then evaluate its performance on MNIST test dataset [pretrained-MNIST](https://github.com/arkya-art/MIDAS-Task-2021/blob/master/Task%202.2/pretrained-MNIST.py)

#### checkpoint_mnist_pretrained
Contains the checkpoint of the model in pretrained-MNIST.py [checkpoint_mnist_pretrained](https://github.com/arkya-art/MIDAS-Task-2021/tree/master/Task%202.2/checkpoint_mnist_pretrained)


## Task 2.3
#### main.py
The main objective of this file is to train the pretrained model from Task 2.1 on MNIST train dataset and
evaluate it on the MNIST test dataset. [main](https://github.com/arkya-art/MIDAS-Task-2021/blob/master/Task%202.3/main.py)

#### checkpoint 
Contains the checkpoint of the model in main.py [checkpoint](https://github.com/arkya-art/MIDAS-Task-2021/tree/master/Task%202.3/checkpoint_mnist_pretrained2.1)

* train



* .gitignore



* experiment log
This contains the detailed information about the processes adopted while solving the given tasks. [Experiment-Log](https://github.com/arkya-art/MIDAS-Task-2021/blob/master/experiment%20log.ipynb)

# Model Task 2.1

## Proposed Work
In the proposed work we followed a step wise procedure to design our model,the first step involves the data preprocessing step.

### Data Preprocessing
Dataset is taken and firstly passed through a data-generator function which performs series of steps. Firstly the input grayscale image is normalized, the normalization of image data is a key process which 
that ensures that each input parameter (pixel, in this case) has a same data distribution, this also helps in faster convergence of the loss function, as suggested within this [article](https://becominghuman.ai/image-data-pre-processing-for-neural-networks-498289068258)
This step is followed by Data augmentation techniques, this step helps to reduce the variance problem. There are many ways by which you perform the [data augmentation](https://nanonets.com/blog/data-augmentation-how-to-use-deep-learning-when-you-have-limited-data-part-2/) technique, some of the effective ways are image rotation, zooming of image, width and height shift.
This were the common techniques which were common in many research paper related to image recognition technique.  

### Related Works
[Ankur AUS](http://users.cecs.anu.edu.au/~Tom.Gedeon/conf/ABCs2018/paper/ABCs2018_paper_175.pdf) In this paper, different classifier algorithms are used in this model to check the performance of the model.
Consideration is made for various hyperparameters, and the convolutional neural network by taking account of the image
length and width to make a stride. The network also comprises pooling layers, and max-pooling is applied to the network.

[Chen et. al.](https://arxiv.org/pdf/1811.08278.pdf) In this paper the author comapers the 4 different types of neural network on the very famous MNIST dataset. They used a CNN based architecture alongwith its modifies version, Deep residual network, the dense convolutional network and the 
Capsule Network. It was showed that CapsNet perform best across datasets.It was found that CapsNet requires only a small amount of data to achieve excellent performance.


[Cho et. al.](https://www.researchgate.net/publication/326152300_Comparisons_of_Deep_Learning_Algorithms_for_MNIST_in_Real-Time_Environment) They have managed to implement out various models some of which include apsule network, deep residual learning model, convolutional neural network and multinomial logistic regression to recognize handwritten digits in real time environment

### Model Architecture
After going through various research papers and taking into account the amount of data we have had to train this model. I have created a baseline model using a simple CNN architecture with carefully selected hyperparameters as mentioned in this [article](https://towardsdatascience.com/hyper-parameter-tuning-techniques-in-deep-learning-4dad592c63c8). 
Taken the input image of 28-28-1 dimension and then applied two back to back convolutional network with 64 3-3 filter followed by ReLu activation. Convolution layer will compute the output of neurons that are connected to local regions in the input, each computing a dot
product between their weights and a small region they are connected to in the input volume and ReLu helps in acheiving the non-linear properties. Then it is followed by MaxPooling and batch normalization(BN) where BN helps in normalizing each batch correspondingly affecting the convergence rate.The same block of layer is again followed but this time using 128 3-3 filters and
then followed by 256 3-3  filters then flatten it and then connected with Fully connected layers to extract out the features. The model defined in this [Research Paper](https://ijrar.org/papers/IJRAR1903931.pdf) resembles my model but differs in the dataset used, they used the EMINIST dataset alongwith their own generated data which gives their model a very good performance.
Also the [Resarch Paper](https://www.mdpi.com/1424-8220/20/12/3344) represents the use of 3 Layer and a 4 layer CNN model alogwith different set cases of Hyper Parameter tuning, the results shows that a 3 Layer model performs well with 12-24-32 feature map and a 4 layer model performs well with 12-24-28-32 feature map.

### Model Performance
The model has 84.91% accuracy on the training dataset and about 54.17% accuracy in validation dataset

This shows the ![Result](https://github.com/arkya-art/MIDAS-Task-2021/blob/master/Images/model%20performance%202.1.jpg)
### Conclusion
There exist large variance error due to overfitting of the dataset, I have tried to apply many possible techniques but could not further improve the performance of the model 

# Model Task 2.2

In this we are comparing both the models formed in Task 2.2 

### Pretrained Model and Randomly Initialized model on MNIST Dataset

This model was trained for 10 epochs in which the train data has 938 steps per epoch and the validation dataset has 157 steps per epoch.
I obtained a training accuracy of 99.68% after 10 epochs while on validation dataset I obtained an accuracy of 99.3%. I can conclude from 
this data that the pretrained model performs very well in terms of bias i.e. fitting the training data and in terms of variance i.e not overfitting
the training dataset as resulted from a very good accuracy in validation dataset.

Whereas in the randomly initialized model, it was trained for 10 epochs in which the train data has 938 steps per epoch and the validation dataset has 157 steps per epoch which is same as earlier model.
We obtained a training accuracy of 99.57% after 10 epochs while on validation dataset I obtained an accuracy of 98.87%.

On comparing both the models in this aspect the pretrained model performed pretty well.

[pretrained model accuracy]()

[Randomly initialized model accuracy]()

On Comparing the epoch loss and accuracy for both the model, the randomly initialized model saturates earlier as compared to the pretrained model on 0-9 dataset. The graphs can be shown below:-

[pretrained model]()

[Randomly initialized model]()

On comparing the confusion matrix for both the models it can be found that there are 113 misclassified images in randomly initialized model and most of the misclassification lies in the upper right diagonal
whereas in the pretrained model there exists 70 misclassified images and the most of the misclassification lies in the lower left diagonal. It can be concluded that for randomly initialized model it is misclassifying
for the larger number mostly (i.e > 5) and for pretrained model it is misclassifying the lower end numbers
 
[pretrained model confusion matrix]()

[Randomly initialized model confusion matrix]() 
 
# Model Task 2.3

### NOTE
### I was not able to download the dataset for the Task 2.3 and so I contacted the coordinator of Task2 and he suggested me to work with the inbuilt MNIST dataset instead of the provided dataset and to present my results according to that.

The pretrained model from Task 2.1 was trained in the MNIST datset, This model was trained for 10 epochs in which the train data has 938 steps per epoch and the validation dataset has 157 steps per epoch.
I obtained a training accuracy of 99.62% after 10 epochs while on validation dataset I obtained an accuracy of 99.21%. I can conclude from 
this data that the pretrained model performs very well in terms of bias i.e. fitting the training data and in terms of variance i.e not overfitting
the training dataset as resulted from a very good accuracy in validation dataset.

According to the earlier model (randomly initialized model) which was trained and validated on MNIST test data, Obtained a training accuracy of 99.57% after 10 epochs while on validation dataset it obtained an accuracy of 98.87%
So the pretrained model performs better as compared to the pre-trained dataset

[pretrained model]()

[pretrained model confusion matrix]()