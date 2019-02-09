## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # ---------------------------------
        # Layer Details:
        # ---------------------------------
        
        
        # LAYER 1
        # Convolution2d1 32  (4x4)
        self.conv_d1 = nn.Conv2d(1, 32, 4)
        self.pool_d1 = nn.MaxPool2d(2, 0)
        self.drop_d1 = nn.Dropout(p=0.1)
        
        # LAYER 2
        # Convolution2d2 64  (3x3)
        self.conv_d2 = nn.Conv2d(32, 64, 3)
        self.pool_d2 = nn.MaxPool2d(2, 0)
        self.drop_d2 = nn.Dropout(p=0.2)
        
        # LAYER 3
        # Convolution2d3 128 (2x2)
        self.conv_d3 = nn.Conv2d(64, 128, 2)
        self.pool_d3 = nn.MaxPool2d(2, 0)
        self.drop_d3 = nn.Dropout(p=0.3)
        
        # LAYER 4
        # Convolution2d4 256 (1x1)   
        self.conv_d4 = nn.Conv2d(128, 256, 1)
        self.pool_d4 = nn.MaxPool2d(2, 0)
        self.drop_d4 = nn.Dropout(p=0.4)
        
        self.drop_d5 = nn.Dropout(p=0.5)
        self.drop_d6 = nn.Dropout(p=0.6)
        
        self.fc1 = nn.Linear(43264, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 136)
        
        
        
    def forward(self, x):
        # Layer 1
        x = self.pool_d1(F.relu(self.conv_d1(x)))
        x = self.drop_d1(x)


        # Layer 2
        x = self.pool_d2(F.relu(self.conv_d2(x)))
        x = self.drop_d2(x)

        # Layer 3
        x = self.pool_d3(F.relu(self.conv_d3(x)))
        x = self.drop_d3(x)

         # Layer 4
        x = self.pool_d4(F.relu(self.conv_d4(x)))
        x = self.drop_d4(x)

        # Flatten
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.drop_d5(x)
        
        x = F.relu(self.fc2(x))
        x = self.drop_d6(x) 
     
        x = self.fc3(x)

        return x