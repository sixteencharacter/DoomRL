import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models
from env import device , n_actions
from torchinfo import summary
from icecream import ic
from config import *

# Baseline from 1605.02097 (https://arxiv.org/pdf/1605.02097)
class CNN(nn.Module) :

    def __init__(self,n_actions) :
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,7)
        self.conv2 = nn.Conv2d(32,32,4)
        self.maxpool = nn.MaxPool2d(2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.ff = nn.Linear(3072,800)
        self.ff2 = nn.Linear(800,n_actions)
        
    
    def forward(self,x) : 
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = x.flatten(1)
        x = F.relu(self.ff(x))
        x = self.ff2(x)
        return x

class ResNet(nn.Module):

    def __init__(self, n_actions):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT
        self.backbone = models.resnet18(weights=weights)
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(feat_dim, n_actions)

    def forward(self, x):
        # We now expect x to be pre-resized to 112x112 or similar by the preprocessor
        x = self.backbone(x)
        return self.head(x)

class ActorCriticResNet(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT
        self.backbone = models.resnet18(weights=weights)
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.actor = nn.Linear(feat_dim, n_actions)
        self.critic = nn.Linear(feat_dim, 1)

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224))
        x = self.backbone(x)
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

class ActorCriticCNN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 7)
        self.conv2 = nn.Conv2d(32, 32, 4)
        self.maxpool = nn.MaxPool2d(2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.ff = nn.Linear(3072, 800)
        
        self.actor = nn.Linear(800, n_actions)
        self.critic = nn.Linear(800, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = x.flatten(1)
        x = F.relu(self.ff(x))
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

def create_q_network(arch, n_actions):
    if arch == "Baseline":
        if METHOD == "PPO":
            return ActorCriticCNN(n_actions=n_actions)
        return CNN(n_actions=n_actions)
    if arch == "ResNet":
        if METHOD == "PPO":
            return ActorCriticResNet(n_actions=n_actions)
        return ResNet(n_actions=n_actions)
    raise ValueError(f"Unsupported ARCH: {arch}")

if __name__ == "__main__" :
    model = None
    try :
        model = create_q_network("Baseline", n_actions=2 ** (n_actions))
        print("All model compiled successfully")
    except :
        print(summary(model,(1,3,120,160)))
        print("Error occurred in some model")