import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DQNNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQNNetwork, self).__init__()
        self.input_shape = input_shape
        
        if len(input_shape) == 3 and input_shape[2] == 3: 
            channels = input_shape[2]
            height = input_shape[0]
            width = input_shape[1]
        else:  
            channels = input_shape[0]
            height = input_shape[1] 
            width = input_shape[2]
        
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        conv_out_size = self._get_conv_out_size((channels, height, width))
        
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, n_actions)
    
    def _get_conv_out_size(self, shape):
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))
    
    def forward(self, x):
        if len(x.shape) == 5:  
            x = x.permute(0, 1, 4, 2, 3) 
            x = x.reshape(x.size(0), -1, x.size(3), x.size(4))  
        elif len(x.shape) == 4 and x.size(3) == 3: 
            x = x.permute(0, 3, 1, 2)  
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x)) 
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DuelingDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DuelingDQN, self).__init__()
        
        if len(input_shape) == 3 and input_shape[2] == 3: 
            channels = input_shape[2]
            height = input_shape[0]
            width = input_shape[1]
        else:  
            channels = input_shape[0]
            height = input_shape[1]
            width = input_shape[2]
        
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        conv_out_size = self._get_conv_out_size((channels, height, width))
        
        self.value_fc = nn.Linear(conv_out_size, 512)
        self.value = nn.Linear(512, 1)
        
        self.advantage_fc = nn.Linear(conv_out_size, 512)
        self.advantage = nn.Linear(512, n_actions)
        
    def _get_conv_out_size(self, shape):
        """Calculate conv output size for given input shape"""
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))
    
    def forward(self, x):
        if len(x.shape) == 5:  
            x = x.permute(0, 1, 4, 2, 3)  
            x = x.reshape(x.size(0), -1, x.size(3), x.size(4))  
        elif len(x.shape) == 4 and x.size(3) == 3:  
            x = x.permute(0, 3, 1, 2)  
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        value = F.relu(self.value_fc(x))
        value = self.value(value)
        
        advantage = F.relu(self.advantage_fc(x))
        advantage = self.advantage(advantage)
        
        return value + advantage - advantage.mean(dim=1, keepdim=True)