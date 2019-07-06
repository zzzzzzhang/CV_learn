
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


# In[ ]:


# change color
def random_change_color(img):
    b,g,r = cv2.split(img)
    def color_process(channel):
        channel = channel.astype('uint32')
        rand = random.randint(-50,50)
#         uint8 no more than 255
        channel = channel + rand
        channel[channel > 255] = 255
        channel[channel < 0] = 0
        channel = channel.astype(img.dtype)
        return channel
    b = color_process(b)
    g = color_process(g)
    r = color_process(r)
    return cv2.merge((b, g, r))


# In[ ]:


img_random_color = random_change_color(img_cat)
cv2.imshow('img_random_color', img_random_color)
hotKey()


# In[ ]:


# gamma correction
def gamma_correct(img,gamma):
    table = []
    invGamma = 1.0/gamma
    for i in range(256):
        table.append(((i/255.0)**invGamma) * 255)
    table = np.array(table).astype('uint8')
    return cv2.LUT(img,table)


# In[ ]:


img_gamma_correct = gamma_correct(img_cat,3)
cv2.imshow('img_gamma_correct', img_gamma_correct)
hotKey()


# In[ ]:


# histogram for gray image
cv2.imshow('cat_gray',img_cat_gray)
img_cat_gray_histed = cv2.equalizeHist(img_cat_gray)
cv2.imshow('img_cat_gray_histed',img_cat_gray_histed)
hotKey()


# In[ ]:


# histogram for rbg image
cv2.imshow('cat',img_cat)
b,g,r = cv2.split(img_cat)
b = cv2.equalizeHist(b)
g = cv2.equalizeHist(g)
r = cv2.equalizeHist(r)
img_cat_histed = cv2.merge((b,g,r))
cv2.imshow('img_cat_histed',img_cat_histed)
hotKey()


# In[ ]:


# histogram for YUV image
# y: luminance(明亮度), u&v: 色度饱和度
cv2.imshow('img_cat',img_cat)
img_cat_yuv = cv2.cvtColor(img_cat,cv2.COLOR_BGR2YUV)
cv2.imshow('img_cat_yuv',img_cat_yuv)
channel_yuv = cv2.split(img_cat_yuv)
channel_yuv[0] = cv2.equalizeHist(channel_yuv[0])
img_cat_yuv = cv2.merge(channel_yuv)
img_cat_yuv2bgr = cv2.cvtColor(img_cat_yuv,cv2.COLOR_YUV2BGR)
cv2.imshow('img_cat_yuv_histed',img_cat_yuv2bgr)
hotKey()


# In[ ]:


# histogram plot
img_small_cat = cv2.resize(img_cat, (img_cat.shape[0]//2, img_cat.shape[1]//2))
plt.hist(img_cat.flatten(), 256, [0, 256], color = 'r')
img_yuv = cv2.cvtColor(img_small_cat, cv2.COLOR_BGR2YUV)
# equalize the histogram of the Y channel
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])   # only for 1 channel
# convert the YUV image back to RGB format
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)   # y: luminance(明亮度), u&v: 色度饱和度
cv2.imshow('Color input image', img_small_cat)
cv2.imshow('Histogram equalized', img_output)
hotKey()


# In[ ]:


# rotation
M = cv2.getRotationMatrix2D((img_cat.shape[1]//2, img_cat.shape[0]//2), 45, 1) #width，height，angle，scale
img_cat_rotate = cv2.warpAffine(img_cat, M, (img_cat.shape[1],img_cat.shape[0]))
cv2.imshow('img_cat_rotate',img_cat_rotate)
hotKey()


# In[ ]:


# Affine transform
height,width,channels = img_cat.shape
points_1 = np.float32([[0, 0], [width - 1, 0], [0, height - 1]])
points_2 = np.float32([[width * 0.2, height * 0.1], [width * 0.9, height * 0.2], [width * 0.1, height * 0.9]])
M = cv2.getAffineTransform(points_1,points_2)
img_cat_affined = cv2.warpAffine(img_cat, M, (width,height))
cv2.imshow('img_cat_affined',img_cat_affined)
hotKey()


# In[ ]:


# perspective transform
def random_warp(img,random_num = 50):
    '''
    random perspective transform
    '''
    height, width, channels = img.shape
    #get warp points
    
    #origin points

    x1 = random.randint(0, random_num)
    y1 = random.randint(0, random_num)
    
    x2 = random.randint(0, random_num)
    y2 = random.randint(-random_num + height - 1, height - 1)
    
    x3 = random.randint(-random_num + width - 1, width - 1)
    y3 = random.randint(0, random_num)
    
    x4 = random.randint(-random_num + width - 1, width - 1)
    y4 = random.randint(-random_num + height - 1, height - 1)
    
    #target points
    tx1 = random.randint(0, random_num)
    ty1 = random.randint(0, random_num)
    
    tx2 = random.randint(0, random_num)
    ty2 = random.randint(-random_num + height - 1, height - 1)
    
    tx3 = random.randint(-random_num + width - 1, width - 1)
    ty3 = random.randint(0, random_num)
    
    tx4 = random.randint(-random_num + width - 1, width - 1)
    ty4 = random.randint(-random_num + height - 1, height - 1)
    
    points_ori = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    points_tag = np.float32([[tx1, ty1], [tx2, ty2], [tx3, ty3], [tx4, ty4]])
    M_warp = cv2.getPerspectiveTransform(points_ori, points_tag)
    img_Perspected = cv2.warpPerspective(img, M_warp, (width,height))
    return img_Perspected


# In[ ]:


img_cat_prespected = random_warp(img_cat)
cv2.imshow('img_cat_prespected', img_cat_prespected)
hotKey()
