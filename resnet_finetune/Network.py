# coding: utf-8

from torchvision import models
from torch import nn

class Net(nn.Module):
    def __init__(self):
        '''
        初始化一些层，使用resnet18作为backbone
        backbone最后一层输出改变成2
        增加一个softmax激活层
        '''
        super(Net,self).__init__()
        self.resnet18 = models.resnet18(pretrained= True)
        self.resnet18.fc = nn.Linear(in_features=512, out_features=2, bias=True)
#         self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        '''
        传播过程
        '''
        #输入尺寸为224*224
        x = self.resnet18(x)
#         x = self.softmax(x)
        return x