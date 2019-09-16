
# coding: utf-8

import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 8, 5, 2, 0)
        
        self.conv2_1 = nn.Conv2d(8, 16, 3, 1, 0)
        self.conv2_2 = nn.Conv2d(16, 16, 3, 1, 0)
        
        self.conv3_1 = nn.Conv2d(16, 24, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(24, 24, 3, 1, 0)
        
        self.conv4_1 = nn.Conv2d(24, 40, 3, 1, 1)
        self.conv4_2 = nn.Conv2d(40, 80, 3, 1, 1)
        
        self.p1 = nn.Linear(4 * 4 * 80, 128)
        self.p2 = nn.Linear(128, 128)
        self.p3 = nn.Linear(128, 42)
        
        self.prelu = nn.PReLU()
        
        self.ave_pool = nn.AvgPool2d(2, 2, ceil_mode= True)
    
    def forward(self, x):
        x = self.ave_pool(self.prelu(self.conv1_1(x)))
        
        x = self.ave_pool(self.prelu(self.conv2_2(self.prelu(self.conv2_1(x)))))
        
        x = self.ave_pool(self.prelu(self.conv3_2(self.prelu(self.conv3_1(x)))))
        
        x = self.prelu(self.conv4_2(self.prelu(self.conv4_1(x))))
        
        x = x.view(-1, 4 * 4 * 80)
        
        p1 = self.prelu(self.p1(x))
        
        p2 = self.prelu(self.p2(p1))
        
        p3 = self.p3(p2)
        
        return p3

