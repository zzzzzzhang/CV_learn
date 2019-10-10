# coding: utf-8

import numpy as np
import cv2
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image, ImageDraw 
import itertools
import random

def channel_norm(img):
    mean = np.mean(img)
    std = np.std(img)
    pixels = (img-mean)/(std)
    return pixels

def parse_line(line):
    line_parts = line.tolist()
    img_name = line_parts[0]
    #rect = list(map(int, list(map(float, line_parts[1:5]))))
    rect = list(map(int, line_parts[1:5]))
    landmarks = list(map(float, line_parts[5: len(line_parts)]))
    return img_name, rect, landmarks

class Normalize(object):
    """
        Resieze to train_boarder x train_boarder. Here we use 112 x 112
        Then do channel normalization: (image - mean) / std_variation
    """
    def __init__(self,train_boarder= 112):
        self.train_boarder = train_boarder
    def __call__(self,sample):
        img, landmask = sample['image'], sample['landmarks']
        img_resize = np.asarray(img.resize((self.train_boarder, self.train_boarder),Image.BILINEAR))
#         img_resize = channel_norm(img_resize)
        return {'image':img_resize,'landmarks':landmask}
        
class  FlipHorizontal(object):
    '''
    flip horizontal
    '''
    def __init__(self,p= 0.5, train_boarder= 112):
        self.p = p
        self.train_boarder = train_boarder
    def __call__(self,sample):
        img, landmask = sample['image'], sample['landmarks']
        if random.random() < self.p:
            img = img[:,::-1].copy()
            landmask[0::2] = self.train_boarder - landmask[0::2].copy()
        return {'image':img,'landmarks':landmask}

class  RandomRotation(object):
    '''
    RandomRotation(0,15)
    '''
    def __init__(self, train_boarder= 112, p= 0.5):
        self.train_boarder = train_boarder
        self.p = p
    def __call__(self,sample):
        img, landmask = sample['image'], sample['landmarks']
        if random.random() < self.p:
            ang = random.randint(-5, 5)
            scale = 1.0
            M = cv2.getRotationMatrix2D((self.train_boarder/2, self.train_boarder/2), ang, scale)
            img = cv2.warpAffine(img, M, (self.train_boarder,self.train_boarder), flags= cv2.INTER_LINEAR)
            xs = landmask[::2].copy()
            ys = landmask[1::2].copy()
        
            #opencv获得的旋转矩阵是调整过的，需要注意
            mxy = (np.c_[xs,ys] - np.array([self.train_boarder/2, self.train_boarder/2])) 
            xys = (mxy.dot( np.transpose( M[:,:2] ) ) + np.array([self.train_boarder/2, self.train_boarder/2]))
        
            landmask[::2] = xys[:,0]
            landmask[1::2] = xys[:,1]
        return {'image':img,'landmarks':landmask}

class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        Tensors channel sequence: N x C x H x W
    """
    def __call__(self,sample):
        '''
        numpy img: H*W*C
        torch.tensorimg: N*C*H*W
        '''
        img, landmarks = sample['image'], sample['landmarks']
#         如果不是灰度图要改变维度
#         img = img.transpose((2, 0, 1))
#         img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=0)
        return {'image':torch.from_numpy(img).float(), 
                'landmarks':torch.from_numpy(landmarks).float()}

class FaceLandmarksDataset():
    def __init__(self, data, transforms= None, train_boarder= 112):
        '''
        :param lines: src_line
        :param transform: data transform
        '''
        self.data = data
        self.transforms = transforms
        self.train_boarder = train_boarder
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, rect, landmarks = parse_line(self.data.values[idx])
        #转为灰度图
        img = Image.open(img).convert('L')
        img = img.crop(tuple(rect))
        landmarks = np.array(landmarks).astype('float64')
        
        #对lanmarks做变换
        w = rect[2] - rect[0]
        h = rect[3] - rect[1]
        
        k_w = self.train_boarder/w
        k_h = self.train_boarder/h
        
        landmarks[::2] *= k_w
        landmarks[1::2] *= k_h
        
        sample = {'image':img, 'landmarks':landmarks}
        sample = self.transforms(sample)
        return sample

def load_data(filepath, phase):
    '''
    加载数据
    '''
    df = pd.read_csv(filepath)
    if phase == 'Train' or phase == 'train':
        tsfm = transforms.Compose([
            Normalize(),                # do channel normalization
            FlipHorizontal(),           # do Flip Horizontal
            RandomRotation(),           # do Random Rotation
            ToTensor()]                 # convert to torch type: NxCxHxW
        )
    else:
        tsfm = transforms.Compose([
            Normalize(),
            ToTensor()]
        )
    data_set = FaceLandmarksDataset(df, transforms= tsfm)
    return data_set

def get_train_test_set():
    train_set = load_data('F:/data/cv_learn/项目/项目二/train/train_annotation.csv','train')
    valid_set = load_data('F:/data/cv_learn/项目/项目二/test/test_annotation.csv','test')
    return train_set, valid_set

def drawLandMarks(path, idx):
    '''
    在resize后的图上画出landmarks
    '''
    dataset = load_data(path, 'train')
    
    sample = dataset[idx]
    img = transforms.ToPILImage()(sample['image'].type(torch.uint8))
    landmarks = sample['landmarks'].numpy()
    xs = landmarks[::2]
    ys = landmarks[1::2]
    
    draw = ImageDraw.Draw(img)
    draw.point(list(zip(xs,ys)),fill = (0))
    img.show()

if __name__ == '__main__':
    path= 'F:/data/cv_learn/项目/项目二/train/train_annotation.csv'
    drawLandMarks(path= path, idx= 1564)
