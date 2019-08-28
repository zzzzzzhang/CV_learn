# coding: utf-8

import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,3,3)
        self.maxpool1 = nn.MaxPool2d(kernel_size= 2)
        self.relu1 = nn.ReLU(inplace= True)
        
        self.conv2 = nn.Conv2d(3,6,3)
        self.maxpool2 = nn.MaxPool2d(kernel_size= 2)
        self.relu2 = nn.ReLU(inplace= True)
        
        self.fc1_1 = nn.Linear(6 * 30 * 30, 150)
        self.relu3_1 = nn.ReLU(inplace= True)
        self.fc1_2 = nn.Linear(6 * 30 * 30, 150)
        self.relu3_2 = nn.ReLU(inplace= True)
        
        self.drop_1 = nn.Dropout(p = 0.5)
        self.drop_2 = nn.Dropout(p = 0.5)
        
        self.fc2_1 = nn.Linear(150,2)
        self.fc2_2 = nn.Linear(150,3)
        
        self.softmax_1 = nn.Softmax(dim = 1)
        self.softmax_2 = nn.Softmax(dim = 1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.relu2(x)
        
        x = x.view(-1, 6 * 30 * 30)
        
        x_class = self.fc1_1(x)
        x_class = self.relu3_1(x_class)
        x_class = self.drop_1(x_class)
        
        x_class = self.fc2_1(x_class)
        x_class = self.softmax_1(x_class)
        
        x_species = self.fc1_2(x)
        x_species = self.relu3_2(x_species)
        x_species = self.drop_2(x_species)
        
        x_species = self.fc2_2(x_species)
        x_species = self.softmax_2(x_species)
        return x_class, x_species

