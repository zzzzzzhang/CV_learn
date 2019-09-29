
# coding: utf-8

import os
import pandas as pd
from PIL import Image, ImageDraw 


def make_anno(path_root):
    '''
    生成截取后的训练样本和测试样本
    '''
    path_pictures = path_root + 'pictures/'
    path_labels = path_root + 'labels/label.txt'
    
    os.makedirs(path_root + 'train',exist_ok= True)
    os.makedirs(path_root + 'test',exist_ok= True)
    train = path_root + 'train/train_annotation.csv'
    test = path_root + 'test/test_annotation.csv'
    
    dataIfo = {'path':[], 'bx1':[], 'by1':[], 'bx2':[], 'by2':[]}
    for j in range(21):
        dataIfo['x' + str(j)] = []
        dataIfo['y' + str(j)] = []

    labels = pd.read_csv(path_labels, header= None, delimiter= ' ').values
    for i in range(len(labels)):
        imgpath = os.path.join(path_pictures,labels[i,0])
        img = Image.open(imgpath)
        x1 = labels[i,1]
        y1 = labels[i,2]
        x2 = labels[i,3]
        y2 = labels[i,4]
        
        expand = 0.2
        x1 = round(x1 - (x2 - x1) * expand if x1 - (x2 - x1) * expand >= 0 else 0)
        x2 = round(x2 + (x2 - x1) * expand if x2 + (x2 - x1) * expand <= img.width else img.width)
        y1 = round(y1 - (y2 - y1) * expand if y1 - (y2 - y1) * expand >= 0 else 0)
        y2 = round(y2 + (y2 - y1) * expand if y2 + (y2 - y1) * expand <= img.height else img.height)
        
#         img = img.crop((x1, y1, x2, y2))
        dataIfo['path'].append(imgpath)
        dataIfo['bx1'].append(x1)
        dataIfo['by1'].append(y1)
        dataIfo['bx2'].append(x2)
        dataIfo['by2'].append(y2)
        for j in range(21):
            idx = j*2 + 5
            x_idx = labels[i,idx]
            y_idx = labels[i,idx + 1]
            dataIfo['x' + str(j)].append(round(x_idx - x1,2))
            dataIfo['y' + str(j)].append(round(y_idx - y1,2))
            
    dataIfo = pd.DataFrame(dataIfo)
    dataIfo.iloc[:round(0.7*len(dataIfo)), :].to_csv(train)
    dataIfo.iloc[round(0.7*len(dataIfo)):, :].to_csv(test)
    return pd.DataFrame(dataIfo)


def visualize_dataset(dataIfo, idx):
    '''
    查看数据
    '''
    print('共有{}条数据'.format(len(dataIfo)))
    
    img = Image.open(dataIfo['path'][idx])
#     img.show()
    x1 = dataIfo['bx1'][idx]
    y1 = dataIfo['by1'][idx]
    x2 = dataIfo['bx2'][idx]
    y2 = dataIfo['by2'][idx]
#     print(img.size)
    img = img.crop((x1, y1, x2, y2))
    draw = ImageDraw.Draw(img)
    
    xs = dataIfo.values[idx,5::2]
    ys = dataIfo.values[idx,6::2]

    assert len(xs) == len(ys)

    draw.point(list(zip(xs,ys)),fill = (255, 0, 0))
    img = img.resize((256,256),Image.ANTIALIAS)
    
    img.show()


if __name__ == '__main__':
    path_root = 'D:/cv_learn/projectII/'        
    dataIfo = make_anno(path_root)
    visualize_dataset(dataIfo, 2591)

