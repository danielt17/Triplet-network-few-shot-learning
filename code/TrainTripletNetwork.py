# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 15:08:36 2021

@author: danie
"""

# %% Imports

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from barbar import Bar

from DataBase import FashionMNIST_t
from Losses import tripletLoss, CustomLoss
from TripletNetwork import TripletNetModel

# %% Functions

def LoadData(batch_size):
    trainset = FashionMNIST_t('../data', train=True, download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914,),(0.2023,))]))
    testset = FashionMNIST_t('../data', train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914,),(0.2023,))]))
    train_loader = torch.utils.data.DataLoader(trainset,batch_size = batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset,batch_size = batch_size ,shuffle=True)
    return train_loader, test_loader

def lossFunction(dist_plus, dist_minus, embedded_anchor, embedded_positive, embedded_negative,loss_type):
    if loss_type == 0:
        loss = tripletLoss(anchor,positive,negative)
    elif loss_type == 1:
        loss = CustomLoss(dist_plus,dist_minus)
    return loss

# %% Main

if __name__ == '__main__':
    lr = 5e-5; 
    batch_size = 64;
    epochs = 100;
    gamma = 0.9999;
    loss_type = 1; # 0 - Triplet loss, 1 - Custom loss paper
    save_model = True;
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = LoadData(batch_size)
    model = TripletNetModel(device)
    optimizer = Adam(model.parameters(),lr = lr)
    scheduler = ExponentialLR(optimizer, gamma=gamma)
    losses = []; dists_plus = []; dists_minus = []
    plt.figure()
    for epoch in range(epochs):
        print('Epoch number: ' + str(epoch))
        plt.clf()
        dists_plus_temp = 0; dists_minus_temp = 0
        for batch_idx, (anchor, negative, positive) in enumerate(Bar(train_loader)):
            anchor = anchor.to(device).requires_grad_()
            negative = negative.to(device).requires_grad_()
            positive = positive.to(device).requires_grad_()
            optimizer.zero_grad()
            # The order is flipped on purpses
            dist_plus, dist_minus, embedded_anchor, embedded_positive, embedded_negative = model(anchor,positive,negative)
            loss = lossFunction(dist_plus, dist_minus, embedded_anchor, embedded_positive, embedded_negative,loss_type=loss_type)
            loss.backward()
            optimizer.step()
            dists_plus_temp =+ np.mean(dist_plus.detach().cpu().numpy())
            dists_minus_temp =+ np.mean(dist_minus.detach().cpu().numpy())
        scheduler.step()
        losses.append(loss.item())
        dists_plus.append(dists_plus_temp); dists_minus.append(dists_minus_temp); 
        plt.subplot(1,3,1)
        plt.plot(losses)
        plt.xlabel('Iteartion [#]')
        plt.ylabel('Loss')
        plt.title('Loss')
        plt.subplot(1,3,2)
        plt.plot(dists_plus)
        plt.xlabel('Iteartion [#]')
        plt.ylabel('MSE')
        plt.title('Similarity')
        plt.subplot(1,3,3)
        plt.plot(dists_minus)
        plt.xlabel('Iteartion [#]')
        plt.ylabel('MSE')
        plt.title('Disimilarity')
        plt.suptitle('Iteration number: ' + str(epoch + 1))
        plt.show()
        plt.pause(0.02)
    if save_model:
        torch.save(model.state_dict(), '../models/TripletModel')
        model.load_state_dict(torch.load('../models/TripletModel'))
        model.eval()
    
    
    
    
