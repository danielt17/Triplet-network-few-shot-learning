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
    '''
    Description:
        This function create train and test data-loaders of triplet tuples.
    Inputs:
        batch_size: batch size to define datasets by
    Returns:
        train_loader: train dataloader class
        test_loader: test dataloader class
    '''
    trainset = FashionMNIST_t('../data', train=True, download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914,),(0.2023,))]))
    testset = FashionMNIST_t('../data', train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914,),(0.2023,))]))
    train_loader = torch.utils.data.DataLoader(trainset,batch_size = batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset,batch_size = batch_size ,shuffle=True)
    return train_loader, test_loader

def lossFunction(dist_plus, dist_minus, embedded_anchor, embedded_positive, embedded_negative,loss_type):
    '''
    Description:
        This function returns the loss value with respect to the loss type, and given netowrk outputs.
    Inputs:
        dist_plus: distance between anchor and positive images
        dist_minus: distance between anchor and negative images
        embedded_anchor: triplet net embbeded vector output of anchor image
        embedded_positive: triplet net embbeded vector output of positive image
        embedded_negative: triplet net embbeded vector output of negative image
        loss_type: 0 for pytorch triplet loss, 1 paper triplet loss
    Returns:
        loss: loss value
    '''
    if loss_type == 0:
        loss = tripletLoss(anchor,positive,negative)
    elif loss_type == 1:
        loss = CustomLoss(dist_plus,dist_minus)
    return loss

def LoadBestModel(load_model,loss_type):
    '''
    Description:
        Loads the best model, with respect to loss type
    Inputs:
        load_model: load learned weights
        loss_type: load weights in correspondes to loss trype
    Returns:
        model: model
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TripletNetModel(device)
    if load_model:
        if loss_type == 0:
            model.load_state_dict(torch.load('../models/triplet_loss/TripletModel'))
        elif loss_type == 1:
            model.load_state_dict(torch.load('../models/custom_loss/TripletModel'))
        model.eval()
    return model

def evaluate_test(test_loader,loss_type):
    '''
    Description:
        Evaluates the model predictive power with respect to test set, and loss type
    Inputs:
        test_loader: test_loader class
        loss_type: 0 for pytorch triplet loss, 1 paper triplet loss
    Returns:
        loss: loss value
    '''
    model.eval()
    losses = []
    for batch_idx, (anchor, negative, positive) in enumerate(Bar(test_loader)):
        anchor = anchor.to(device).requires_grad_()
        negative = negative.to(device).requires_grad_()
        positive = positive.to(device).requires_grad_()
        dist_plus, dist_minus, embedded_anchor, embedded_positive, embedded_negative = model(anchor,positive,negative)
        loss = lossFunction(dist_plus, dist_minus, embedded_anchor, embedded_positive, embedded_negative,loss_type=loss_type)
        losses.append(loss.item())
    losses = np.mean(losses)
    return losses

# %% Main

if __name__ == '__main__':
    # Hyperparameters
    epochs = 500;
    batch_size = 64;
    gamma = 0.99;
    loss_type = 1; # 0 - Triplet loss, 1 - Custom loss paper
    if loss_type == 0:
        lr = 1e-2; 
    elif loss_type == 1:
        lr = 1e-6;
    save_model = False; load_model = False;
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = LoadData(batch_size)
    model = TripletNetModel(device)
    # Optimizer - Adam, Learning rate scheduler - Exponential learning rate
    optimizer = Adam(model.parameters(),lr = lr)
    scheduler = ExponentialLR(optimizer, gamma=gamma)
    losses = []; test_losses = []; dists_plus = []; dists_minus = []
    plt.figure()
    cur_loss = np.inf
    # Training
    for epoch in range(epochs):
        print('Epoch number: ' + str(epoch + 1))
        plt.clf()
        dists_plus_temp = []; dists_minus_temp = []; loss_cur = [];
        for batch_idx, (anchor, negative, positive) in enumerate(Bar(train_loader)):
            anchor = anchor.to(device).requires_grad_()
            negative = negative.to(device).requires_grad_()
            positive = positive.to(device).requires_grad_()
            optimizer.zero_grad()
            # The order is flipped on purpse
            dist_plus, dist_minus, embedded_anchor, embedded_positive, embedded_negative = model(anchor,positive,negative)
            loss = lossFunction(dist_plus, dist_minus, embedded_anchor, embedded_positive, embedded_negative,loss_type=loss_type)
            loss.backward()
            optimizer.step()
            dists_plus_temp_temp = dist_plus.detach().cpu().numpy()
            dists_minus_temp_temp = dist_minus.detach().cpu().numpy()
            norm = np.exp(dists_plus_temp_temp) + np.exp(dists_minus_temp_temp)
            dists_plus_temp.append(np.mean(np.exp(dists_plus_temp_temp)/norm))
            dists_minus_temp.append(np.mean(np.exp(dists_minus_temp_temp)/norm))
            loss_cur.append(loss.item())
        scheduler.step()
        losses.append(np.mean(loss_cur)); test_losses.append(evaluate_test(test_loader,loss_type))
        dists_plus.append(np.mean(dists_plus_temp)); dists_minus.append(np.mean(dists_minus_temp)); 
        print('Train Loss: ' + str(losses[-1]))
        print('Test Loss: ' + str(test_losses[-1]))
        cur_loss = min(test_losses[-1],cur_loss)
        if save_model and cur_loss == test_losses[-1]:
            if loss_type == 0:
                torch.save(model.state_dict(), '../models/triplet_loss/TripletModel')
            elif loss_type == 1:
                torch.save(model.state_dict(), '../models/custom_loss/TripletModel')
        plt.subplot(2,2,1)
        plt.semilogy(losses)
        plt.xlabel('Iteartion [#]')
        plt.ylabel('Loss')
        plt.title('Training loss')
        plt.subplot(2,2,2)
        plt.semilogy(test_losses)
        plt.xlabel('Iteartion [#]')
        plt.ylabel('Loss')
        plt.title('Test loss')
        plt.subplot(2,2,3)
        plt.plot(dists_plus)
        plt.xlabel('Iteartion [#]')
        plt.ylabel('Probability')
        plt.title('Similarity')
        plt.subplot(2,2,4)
        plt.plot(dists_minus)
        plt.xlabel('Iteartion [#]')
        plt.ylabel('Probability')
        plt.title('Dissimilarity')
        plt.suptitle('Iteration number: ' + str(epoch + 1))
        plt.show()
        plt.pause(0.02)
    # Load best model
    model = LoadBestModel(load_model,loss_type)
    
    
    
    
