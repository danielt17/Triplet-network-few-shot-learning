# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 13:59:49 2021

@author: danie
"""

# %% Imports

import torch
import torch.nn as nn
import torch.nn.functional as F

# %% General sub net architecture

class Net(nn.Module):
    '''
    Description:
        Defines backbone CNN
    Returns:
        nn.Module: backbone CNN
        x: network output
    '''
    def __init__(self):
        super(Net, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=50)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    
# %% Triplet net embedding and distances

class TripletNetClass(nn.Module):
    '''
    Description:
        Defines the triplet network class
    Inputs:
        embedding net: backbone network for triplet network (nn.Module)
    Returns:
        nn.Module: triplet network
        dist_a: distance between anchor and positive image
        dist_b: distance between anchor and negative image
        embedded_x / y / z: features vector (vector embedding) of input image anchor/positive/negative  
    '''
    def __init__(self, embeddingnet):
        super(TripletNetClass, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, x, y, z):
        # embeddings
        embedded_x = self.embeddingnet(x)
        embedded_y = self.embeddingnet(y)
        embedded_z = self.embeddingnet(z)
        # distance
        dist_a = F.pairwise_distance(embedded_x, embedded_y, 2)
        dist_b = F.pairwise_distance(embedded_x, embedded_z, 2)
        return dist_a, dist_b, embedded_x, embedded_y, embedded_z

# %% Triplet net model  

def TripletNetModel(device):
    '''
    Description:
        This function creates a triplelet network model on a given device
    Inputs:
        device: cpu or cuda enbaled gpu
    Returns:
        Tnet: Triplet net model on required device
    '''
    model = Net()
    Tnet = TripletNetClass(model).to(device)
    print('Triplet network defined succefully!')
    return Tnet

# %% Main
    
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Tnet = TripletNetModel(device)
    