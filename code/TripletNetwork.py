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
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 20, kernel_size=3)
            self.conv2 = nn.Conv2d(20, 40, kernel_size=3)
            self.conv3 = nn.Conv2d(40, 320, kernel_size=3)
            self.conv3_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 120)
            self.fc2 = nn.Linear(120, 50)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
            x = x.view(x.size(0), 320)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
        
# %% Triplet net embedding and distances

class TripletNetClass(nn.Module):
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
    model = Net()
    Tnet = TripletNetClass(model).to(device)
    print('Triplet network defined succefully!')
    return Tnet

# %% Main
    
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Tnet = TripletNetModel(device)
    