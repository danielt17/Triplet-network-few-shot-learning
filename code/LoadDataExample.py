# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 14:13:19 2021

@author: danie
"""

# %% Imports

import torch
from torchvision import transforms
from DataBase import FashionMNIST_t
import matplotlib.pyplot as plt

# %% Load data
batch_size = 4
trainset = FashionMNIST_t('../data', train=True, download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))
testset = FashionMNIST_t('../data', train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))
train_loader = torch.utils.data.DataLoader(trainset,batch_size = batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset,batch_size = batch_size ,shuffle=True)

def convert_to_imshow_format(image):
    image=image/2+0.5
    image=image.numpy()
    return image.transpose(1,2,0)

dataiter=iter(train_loader)
anchor, negative, positive = dataiter.next()

plt.figure()
for idx,image in enumerate(anchor):
    plt.subplot(3,4,idx+1)
    plt.imshow(convert_to_imshow_format(image)[:,:,0])
    plt.title('Anchor')
for idx,image in enumerate(positive):
    plt.subplot(3,4,idx+5)
    plt.imshow(convert_to_imshow_format(image)[:,:,0])
    plt.title('Positive')
for idx,image in enumerate(negative):
    plt.subplot(3,4,idx+9)
    plt.imshow(convert_to_imshow_format(image)[:,:,0])
    plt.title('Negative')
    
    
