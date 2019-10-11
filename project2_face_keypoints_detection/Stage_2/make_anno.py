
# coding: utf-8

# In[ ]:


import os
import pandas as pd
from PIL import Image, ImageDraw 


# In[ ]:


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
        
        expand = 0.25
        width = x2 - x1 + 1
        height = y2 - y1 + 1
        padding_width = int(width * expand)
        padding_height = int(height * expand)
        
        x1 = round(x1 - padding_width if x1 - padding_width >= 0 else 0)
        x2 = round(x2 + padding_width if x2 + padding_width < img.width else img.width-1)
        y1 = round(y1 - padding_height if y1 - padding_height >= 0 else 0)
        y2 = round(y2 + padding_height if y2 + padding_height < img.height else img.height-1)
        
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
    
    #删除坐标负值样本
    idxs = []
    for i ,row in enumerate(dataIfo.iloc[:,5:].values):
        if all([0 if o < 0 else 1 for o in row]):
            continue
        else:
            idxs.append(i)
    dataIfo = dataIfo.drop(idxs)
    #打乱数据行
    dataIfo = dataIfo.sample(frac= 1, random_state=1)
    dataIfo.iloc[:round(0.9*len(dataIfo)), :].to_csv(train, index= None)
    dataIfo.iloc[round(0.9*len(dataIfo)):, :].to_csv(test, index= None)
    return dataIfo


# In[ ]:


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


# In[ ]:


if __name__ == '__main__':
    path_root = 'data/'        
    dataIfo = make_anno(path_root)
    visualize_dataset(dataIfo, 29)


# In[ ]:


Image.open(dataIfo['path'][29])

