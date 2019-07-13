# coding: utf-8

import numpy as np
import cv2 as cv

def hotKey(key_num = 27):
    key = cv.waitKey()
    if key == key_num:
        cv.destroyAllWindows()

path = r'source/cat.jpg'
img_cat = cv.imread(path,0)

def medianBlur(img, kernel, padding_way = 'zero'):
    '''
    img: a 2-D ndarray
    kernel: a tuple of kernel's shape, must be odd number
    padding_way: str,'same','zero'
    '''
    if (kernel[0] * kernel[1])%2 == 0:
        print('shape of kernel must be odd numbers')
        return None
    img = img_cat.copy()
    w = img.shape[1]
    h = img.shape[0]
    m = kernel[1]
    n = kernel[0]
#     padding method
    if padding_way == 'zero':
        img_padding = np.zeros((h + n//2, w + m//2),dtype='uint8')
        img_padding[n//2 : h + n//2, m//2 : w + m//2] = img.copy()
    elif padding_way == 'same':
        img_padding = np.zeros((h + n//2, w + m//2),dtype='uint8')
        img_padding[n//2 : h + n//2, m//2 : w + m//2] = img.copy()
        img_padding[:n//2,:] == img_padding[n//2,:]
        img_padding[-(n//2):,:] == img_padding[-(n//2) - 1,:]
        img_padding[:,:m//2] == img_padding[:,[m//2]]
        img_padding[:,-(m//2):] == img_padding[:,[-(m//2) - 1]]
    else:
        print('must be same or zero!')
        return None
#convolute
    for i in range(n//2,h):
        for j in range(m//2,w):
#get the box convoluted
            box = img_padding[i - n//2 : i + n//2 + 1,j - m//2 : j + m//2 + 1]
            box = np.sort(box.flatten())
            img[i,j] = box[m*n//2]
    return img

img = medianBlur(img_cat,(5,5),'same')

cv.imshow('img', img)
cv.imshow('img_cat', img_cat)
hotKey()

