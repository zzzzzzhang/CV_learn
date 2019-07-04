
# coding: utf-8

# In[ ]:


import cv2
import random
import numpy as np
from matplotlib import pyplot as plt


# In[ ]:


def hotKey(key_num = 27):
    key = cv2.waitKey()
    if key == key_num:
        cv2.destroyAllWindows()


# In[ ]:


# show img_ori
img_cat = cv2.imread('cat.jpg')
cv2.imshow('cat_ori.jpg',img_cat)
hotKey()


# In[ ]:


# show img_gray
img_cat_gray = cv2.imread('cat.jpg',0)
cv2.imshow('cat_gray.jpg',img_cat_gray)
hotKey()


# In[ ]:


# show the img matrix
print(img_cat)


# In[ ]:


# show img shape
print(img_cat.shape)


# In[ ]:


# show img dtype
print(img_cat.dtype)


# In[ ]:


# img crop 1/2
img_croped = img_cat[0:img_cat.shape[0]//2,0:img_cat.shape[1]//2,:]
cv2.imshow('cat_ori_croped.jpg',img_croped)
hotKey()


# In[ ]:


# color split
b,g,r = cv2.split(img_cat)
cv2.imshow('b',b)
cv2.imshow('g',g)
cv2.imshow('r',r)
hotKey()

