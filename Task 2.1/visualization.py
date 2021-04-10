
"""
Data Visualization Function

****************************  Methodology  *************************************
In this file, I am plotting the images using matplotlib library.
In the data_visualization function, passing the folder path to the images
within the folder path we have many labelled files and within every file we have 
images inside it. 

Plotting with plt.subplot() alogwith the labelled text beneath it

Input - folder path
output - Images

"""
def data_visualization(folder_path):
   
  # importing the libraries  
  from  matplotlib import pyplot as plt
  import random
  import os
  import matplotlib.image as mpimg
  
  # PLotting the figure by figsize, the figsize attribute allows us to specify the width and height of a figure in unit inches
  plt.figure(figsize = (20,20))
  
  for i in range(5):
    file = random.choice(os.listdir(folder_path))
    image_path= os.path.join(folder_path, file)
    img=mpimg.imread(image_path)
    ax=plt.subplot(1,5,i+1)
    ax.title.set_text(file)
    return plt.imshow(img)

