
# coding: utf-8

# In[ ]:


import cv2
import random
import numpy as np


# In[ ]:


img_path = r'source/cat.jpg'
img_cat = cv2.imread(img_path)
pathOut = r'out/'


# In[ ]:


def hotKey(key_num = 27):
    key = cv2.waitKey()
    if key == key_num:
        cv2.destroyAllWindows()


# In[ ]:


class data_augmentation():
    '''
    class_data_augmentation
    '''
    #color shift
    def random_change_color(img, random_num = 50):
        b,g,r = cv2.split(img)
        def color_process(channel):
            channel = channel.astype('uint32')
            rand = random.randint(-random_num,random_num)
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
    
    #rotation
    def rotation(img, angle = 45, scale = 1):
        M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, scale) #width，height，angle，scale
        img_rotate = cv2.warpAffine(img, M, (img.shape[1],img.shape[0]))
        return img_rotate
    
    # perspective transform
    def random_warp(img, random_num = 50):
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
    
    #crop
    def img_crop(img,boxsize = (30,30)):
        random_point_r = random.randint(0, img.shape[0] - 1 - boxsize[0])
        random_point_c = random.randint(0, img.shape[1] - 1 - boxsize[1])
        img_croped = img[random_point_r:random_point_r + boxsize[0], random_point_c:random_point_c + boxsize[1]]
        return img_croped


# In[ ]:


for i in range(10):
    img_processed = data_augmentation.random_change_color(img_cat, random_num= 50)
    img_processed = data_augmentation.rotation(img_processed, angle= 45, scale= 1)
    img_processed = data_augmentation.random_warp(img_processed, random_num= 50)
    img_processed = data_augmentation.img_crop(img_processed,(100,100))
    cv2.imwrite(pathOut + str(i) + '.jpg', img_processed)

