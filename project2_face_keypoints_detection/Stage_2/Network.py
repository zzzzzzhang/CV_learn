
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn


# In[ ]:


class Net_Bn(nn.Module):
    def __init__(self):
        super(Net_Bn, self).__init__()

        avgPool = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # 1*112*112 -> 8*54*54
        self.conv1_1 = nn.Conv2d(1, 8, kernel_size=5, stride=2)
        self.prelu1_1 = nn.PReLU()
        # 8*54*54 -> 8*27*27
        self.pool1 = avgPool

        # 8*54*54 -> 16*25*25
        self.conv2_1 = nn.Conv2d(8, 16, kernel_size=3)
        self.prelu2_1 = nn.PReLU()
        # 16*25*25 -> 16*23*23
        self.conv2_2 = nn.Conv2d(16, 16, kernel_size=3)
        self.prelu2_2 = nn.PReLU()
        # 16*23*23 -> 16*12*12
        self.pool2 = avgPool

        # 16*12*12 -> 24*10*10
        self.conv3_1 = nn.Conv2d(16, 24, kernel_size=3)
        self.prelu3_1 = nn.PReLU()
        # 24*10*10 -> 24*8*8
        self.conv3_2 = nn.Conv2d(24, 24, kernel_size=3)
        self.prelu3_2 = nn.PReLU()
        # 24*8*8 -> 24*4*4
        self.pool3 = avgPool

        # 24*4*4 -> 40*4*4
        self.conv4_1 = nn.Conv2d(24, 40, kernel_size=3, padding=1)
        self.prelu4_1 = nn.PReLU()
        # 40*4*4 -> 80*4*4
        self.conv4_2 = nn.Conv2d(40, 80, kernel_size=3, padding=1)
        self.prelu4_2 = nn.PReLU()


        self.ip1 = nn.Linear(80 * 4 * 4, 128)
        self.preluip1 = nn.PReLU()
        self.ip2 = nn.Linear(128, 128)
        self.preluip2 = nn.PReLU()
        self.bn_ip3 =  nn.BatchNorm1d(128)
        self.ip3 = nn.Linear(128, 42)

    def forward(self, x):
        """
        x: (1,1,112,112)
        retVal: (1, 42)
        """
        x = self.prelu1_1(self.conv1_1(x))
        x = self.pool1(x)

        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = self.pool2(x)

        x = self.prelu3_1(self.conv3_1(x))
        x = self.prelu3_2(self.conv3_2(x))
        x = self.pool3(x)

        x = self.prelu4_1(self.conv4_1(x))
        x = self.prelu4_2(self.conv4_2(x))

        x = x.view(-1, 80 * 4 * 4)
        x = self.preluip1(self.ip1(x))

        x = self.preluip2(self.ip2(x))

        x = self.bn_ip3(x)
        x = self.ip3(x)

        return x


# In[ ]:


class Net_Bn_2d(nn.Module):
    def __init__(self):
        super(Net_Bn_2d, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 16, 3, 1, 1)
        
        self.conv2_1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(16, 32, 3, 1, 1)
        
        self.conv3_1 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(32, 64, 3, 1, 1)
        
        self.conv4_1 = nn.Conv2d(64, 96, 3, 1, 1)
        self.conv4_2 = nn.Conv2d(96, 128, 3, 1, 1)
        
        self.p1 = nn.Linear(14 * 14 * 128, 1024)
        self.p2 = nn.Linear(1024, 256)
        self.p3 = nn.Linear(256, 42)
        
        self.prelu = nn.PReLU()
        
        self.ave_pool = nn.AvgPool2d(2, 2, ceil_mode= True)
        
        self.max_pool = nn.MaxPool2d(2, 2, ceil_mode= True)
        
        self.dropout = nn.Dropout(0.5)
        
#         self.bn2d = nn.BatchNorm2d()
        self.bn1d_1 = nn.BatchNorm1d(14 * 14 * 128)
        self.bn1d_2 = nn.BatchNorm1d(1024)
        self.bn1d_3 = nn.BatchNorm1d(256)
        
    
    def forward(self, x):
        #x:112 * 112 -> 56 * 56
        x = self.max_pool(self.prelu(self.conv1_1(x)))
        #x:56 * 56 -> 28 * 28
        x = self.max_pool(self.prelu(self.conv2_2(self.prelu(self.conv2_1(x)))))
        #x:28 * 28 -> 14 * 14
        x = self.ave_pool(self.prelu(self.conv3_2(self.prelu(self.conv3_1(x)))))
        
        x = self.prelu(self.conv4_2(self.prelu(self.conv4_1(x))))

        x = x.view(-1, 14 * 14 * 128)
        
        x = self.bn1d_1(x)
        
        x = self.dropout(x)

        p1 = self.prelu(self.p1(x))
        
        p2 = self.prelu(self.p2(self.bn1d_2(p1)))
        
        p3 = self.p3(self.bn1d_3(p2))
        
        return p3

