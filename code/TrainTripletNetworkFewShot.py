# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 19:44:17 2021

@author: danie
"""


# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 15:08:36 2021

@author: danie
"""

# %% Imports

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from FewShotLearningDataSet import LoadDataFMnist,CreateTriplets,SupportSetAndQuery
from Losses import CustomLoss
from TripletNetwork import TripletNetModel

# %% Functions

def evaluate_test(Test_X_triplets,TripletTestSize,batch_size):
    model.eval()
    losses = []
    for batch_idx in tqdm(range(TripletTestSize//batch_size)):
        anchor = Test_X_triplets[0][batch_idx*batch_size:(batch_idx+1)*batch_size].to(device).requires_grad_()
        positive = Test_X_triplets[1][batch_idx*batch_size:(batch_idx+1)*batch_size].to(device).requires_grad_()
        negative = Test_X_triplets[2][batch_idx*batch_size:(batch_idx+1)*batch_size].to(device).requires_grad_()
        dist_plus, dist_minus, embedded_anchor, embedded_positive, embedded_negative = model(anchor,positive,negative)
        loss = CustomLoss(dist_plus,dist_minus)
        losses.append(loss.item())
    losses = np.mean(losses)
    return losses

def LoadBestModel():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TripletNetModel(device)
    model.load_state_dict(torch.load('../models/FewShotLearningTripletNetwork/TripletModel'))
    model.eval()
    return model

# %% Main

if __name__ == '__main__':
    epochs = 500;
    batch_size = 64;
    gamma = 0.99;
    lr = 1e-6;
    labels_out = [7,8,9]
    TripletSetSize = 60000
    TripletTestSize = np.int64(60000*0.2)
    k_way=2
    n_shot=50
    train_model = True; FewShotEvaluation = False; 
    save_model = True; load_model = True;
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Train_X,Train_Y,Test_Y,Test_Y,SupportSet_X,SupportSet_Y = LoadDataFMnist(labels_out = labels_out)
    Train_X_triplets, Train_Y_triplets, Train_Index_triplets = CreateTriplets(Train_X,Train_Y,TripletSetSize=TripletSetSize)
    Test_X_triplets, Test_Y_triplets, Test_Index_triplets = CreateTriplets(Train_X,Train_Y,TripletSetSize=TripletTestSize)
    model = TripletNetModel(device)
    optimizer = Adam(model.parameters(),lr = lr)
    scheduler = ExponentialLR(optimizer, gamma=gamma)
    losses = []; test_losses = []; dists_plus = []; dists_minus = []
    plt.figure()
    cur_loss = np.inf
    if train_model:
        for epoch in range(epochs):
            print('Epoch number: ' + str(epoch + 1))
            plt.clf()
            dists_plus_temp = 0; dists_minus_temp = 0; loss_cur = [];
            for batch_idx in tqdm(range(TripletSetSize//batch_size)):
                anchor = Train_X_triplets[0][batch_idx*batch_size:(batch_idx+1)*batch_size].to(device).requires_grad_()
                positive = Train_X_triplets[1][batch_idx*batch_size:(batch_idx+1)*batch_size].to(device).requires_grad_()
                negative = Train_X_triplets[2][batch_idx*batch_size:(batch_idx+1)*batch_size].to(device).requires_grad_()
                optimizer.zero_grad()
                # The order is flipped on purpse
                dist_plus, dist_minus, embedded_anchor, embedded_positive, embedded_negative = model(anchor,positive,negative)
                loss = CustomLoss(dist_plus,dist_minus)
                loss.backward()
                optimizer.step()
                dists_plus_temp =+ np.mean(dist_plus.detach().cpu().numpy())
                dists_minus_temp =+ np.mean(dist_minus.detach().cpu().numpy())
                loss_cur.append(loss.item())
            scheduler.step()
            losses.append(np.mean(loss_cur))
            test_losses.append(evaluate_test(Test_X_triplets,TripletTestSize,batch_size))
            print('Train Loss: ' + str(losses[-1]))
            print('Test Loss: ' + str(test_losses[-1]))
            dists_plus.append(dists_plus_temp); dists_minus.append(dists_minus_temp); 
            cur_loss = min(test_losses[-1],cur_loss)
            if save_model and cur_loss == test_losses[-1]:
                torch.save(model.state_dict(), '../models/FewShotLearningTripletNetwork/TripletModel')
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
            plt.ylabel('MSE')
            plt.title('Similarity')
            plt.subplot(2,2,4)
            plt.plot(dists_minus)
            plt.xlabel('Iteartion [#]')
            plt.ylabel('MSE')
            plt.title('Disimilarity')
            plt.suptitle('Iteration number: ' + str(epoch + 1))
            plt.show()
            plt.pause(0.02)
    if load_model:
        model = LoadBestModel()
    if FewShotEvaluation:
        Query,QueryLabel,classes,SupportSet = SupportSetAndQuery(SupportSet_X,SupportSet_Y,labels_out,k_way=k_way,n_shot=n_shot)
    
    
