# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 14:13:19 2021

@author: danie
"""

# %% Imports

import torch
from torchvision import transforms
from DataBase import FashionMNIST_t

# %% Load data

train_loader = torch.utils.data.DataLoader(FashionMNIST_t('../data', train=True, download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])),shuffle=True)
test_loader = torch.utils.data.DataLoader(FashionMNIST_t('../data', train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])),shuffle=True)