# coding: utf-8

import os
import pandas as pd
from PIL import Image 

roots = 'F:/data/cv_learn/Dataset/'
phase = ['train','val']
classes = ['Mammals', 'Birds']
species = ['rabbits', 'chickens','rats']

dataIfo = {'train':{'path':[],'classes':[],'species':[]},
           'val':{'path':[],'classes':[],'species':[]}}


for p in phase:
    for s in species:
        data_dir = roots + p + '/' + s
        data_filenames = os.listdir(data_dir)
        for filename in data_filenames:
            imgPath = data_dir + '/' + filename
            try:
                img = Image.open(imgPath)
            except OSError:
                pass
            else:
                dataIfo[p]['path'].append(imgPath)
                dataIfo[p]['classes'].append(0 if s in ['rabbits','rats'] else 1)
                dataIfo[p]['species'].append(0 if s == 'rabbits' else 1 if s == 'chickens' else 2)
    annotation = pd.DataFrame(dataIfo[p])
    annotation.to_csv('Classes_{}_annotation.csv'.format(p),index= None)
    print('Classes_{}_annotation is saved'.format(p))


