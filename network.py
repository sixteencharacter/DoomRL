import torch.nn as nn
import torch.nn.functional as F
import torch
from env import device , n_actions
from torchinfo import summary

# Baseline from 1605.02097 (https://arxiv.org/pdf/1605.02097)
class CNN_DQN(nn.Module) :

    def __init__(self,n_actions) :
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,7)
        self.conv2 = nn.Conv2d(32,32,4)
        self.maxpool = nn.MaxPool2d(2)
        self.ff = nn.Linear(14400,800)
        self.ff2 = nn.Linear(800,n_actions)
        
    
    def forward(self,x) : 
        B = x.size(0)
        # x = F.interpolate(x,size=(60,45))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = torch.flatten(x).reshape(B,-1)
        x = F.relu(self.ff(x))
        x = self.ff2(x)
        return x

if __name__ == "__main__" :
    model = None
    try :
        model = CNN_DQN(n_actions=2**(n_actions))
        print("All model compiled successfully")
    except :
        print(summary(model,(1,3,120,160)))
        print("Error occurred in some model")