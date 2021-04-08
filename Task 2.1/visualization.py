
"""
Data Visualization

"""
def data_visualization(folder_path):
    
  from  matplotlib import pyplot as plt
  import random
  import os
  import matplotlib.image as mpimg
  
  
  plt.figure(figsize = (20,20))
  
  for i in range(5):
    file = random.choice(os.listdir(folder_path))
    image_path= os.path.join(folder_path, file)
    img=mpimg.imread(image_path)
    ax=plt.subplot(1,5,i+1)
    ax.title.set_text(file)
    return plt.imshow(img)

