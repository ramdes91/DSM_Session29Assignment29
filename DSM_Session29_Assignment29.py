
# coding: utf-8

# In this assignment students have to compress racoon grey scale image into 5 clusters. In the end, visualize both raw and compressed image and look for quality difference.
# 
# The raw image is available in spicy.misc package with the name face.
# 
# 
# 

# In[1]:


import numpy as np
from sklearn.cluster import KMeans
import scipy.misc
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Visualize the gray scale image

# In[7]:


face = scipy.misc.face(gray=True)
plt.figure(figsize=(10, 3.6))
plt.imshow(face, cmap=plt.cm.gray)
plt.show()


# Compressing the gray scale image into 5 clusters
# 

# In[3]:


rows = face.shape[0]
cols = face.shape[1]
image = face.reshape(rows*cols,1)
kmeans = KMeans(n_clusters = 5)
kmeans.fit(image)


# In[4]:


clusters = np.asarray(kmeans.cluster_centers_) 
labels = np.asarray(kmeans.labels_)  
labels = labels.reshape(rows,cols);
labels


# In[5]:


plt.imsave('compressed_racoon.png',labels);#save compressed image


# Visualize the compressed image

# In[6]:


image = plt.imread('compressed_racoon.png')
plt.figure(figsize=(10, 3.6))
plt.imshow(image)
plt.show()

