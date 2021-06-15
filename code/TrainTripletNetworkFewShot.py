# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 19:44:17 2021

@author: danie
"""

# %% Imports

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.signal import savgol_filter

from FewShotLearningDataSet import LoadDataFMnist,CreateTriplets,SupportSetAndQuery
from Losses import CustomLoss
from TripletNetwork import TripletNetModel

# %% Setup plot

font = {'family' : 'normal',
        'size'   : 14}

matplotlib.rc('font', **font)

# %% Functions

def evaluate_test(Test_X_triplets,TripletTestSize,batch_size):
    '''
    Inputs:
        Test_X_triplets: triplet list (anchor,positive,negative)
        TripletTestSize: test set ssize
        batch_size: batch size
    Returns: 
        losses: loss of test set
    '''
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
    '''
    Returns: 
        model: model with learned weights
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TripletNetModel(device)
    model.load_state_dict(torch.load('../models/FewShotLearningTripletNetwork/TripletModel'))
    model.eval()
    return model

def CalculateAccuracyFewShot(ExpNum,SupportSet_X,SupportSet_Y,labels_out,k_way,n_shot,device):
    '''
    Inputs:
        ExpNum: Number of times to repeat expriement
        SupportSet_X: Support set inputs
        SupportSet_Y: Support set outputs
        labels_out: labels in support set
        k_way: number of classes in actual support set
        n_shot: number of shots in each class in support set
        device: cpu or cuda enbaled gpu 
    Returns: 
        acc: accuracy for given k ways and n shots
        std: accuracy standard devation for given k ways and n shots
    '''
    model.eval()
    acc = []
    for i in range(ExpNum):
        Query,QueryLabel,classes,SupportSet = SupportSetAndQuery(SupportSet_X,SupportSet_Y,labels_out,k_way=k_way,n_shot=n_shot)
        cur_preds = []
        for ls in SupportSet:
            cur_preds.append(model(Query.to(device),ls.to(device),ls.to(device))[0].detach().cpu().numpy())
        cur_preds = np.array(cur_preds)
        cur_preds = np.sum(cur_preds,axis=1)
        pred_label = np.argmin(cur_preds)
        acc.append(100*(pred_label==(len(SupportSet)-1)))
    std = np.std(acc)
    acc = np.mean(acc)
    return acc,std

def CalculateNshotsAccuray(n_shot_num,ExpNum,SupportSet_X,SupportSet_Y,labels_out,k_way,device):
    '''
    Inputs:
        n_shot_num: max number of shots
        ExpNum: Number of times to repeat expriement
        SupportSet_X: Support set inputs
        SupportSet_Y: Support set outputs
        labels_out: labels in support set
        k_way: number of classes in actual support set
        device: cpu or cuda enbaled gpu 
    Returns: 
        acc: (list) accuracy for given k ways and n shots
        std: (list) accuracy standard devation for given k ways and n shots
    '''
    acc = []
    std = []
    for n_shots in tqdm(range(n_shot_num)):
        acc_temp,std_temp = CalculateAccuracyFewShot(ExpNum,SupportSet_X,SupportSet_Y,labels_out,k_way,n_shot = n_shots+1,device=device)
        std.append(std_temp)
        acc.append(acc_temp)
    return acc,std
    
def CalculateKWaysAccuray(n_shot,k_way_num,ExpNum,SupportSet_X,SupportSet_Y,labels_out,device):
    '''
    Inputs:
        n_shot: number of shots
        k_way: maximum number of classes (ways)
        ExpNum: Number of times to repeat expriement
        SupportSet_X: Support set inputs
        SupportSet_Y: Support set outputs
        labels_out: labels in support set
        device: cpu or cuda enbaled gpu 
    Returns: 
        acc: (list) accuracy for given k ways and n shots
        std: (list) accuracy standard devation for given k ways and n shots
    '''
    acc = []
    std = []
    for k_ways in tqdm(range(k_way_num)):
        acc_temp,std_temp = CalculateAccuracyFewShot(ExpNum,SupportSet_X,SupportSet_Y,labels_out,k_way=k_ways+1,n_shot = n_shot,device=device)
        std.append(std_temp)
        acc.append(acc_temp)
    return acc,std

# %% Main

if __name__ == '__main__':
    # Hyperparameters
    epochs = 500;
    batch_size = 64;
    gamma = 0.99;
    lr = 1e-6;
    labels_out = [6,7,8,9]
    TripletSetSize = 60000
    TripletTestSize = np.int64(60000*0.2)
    k_way=4
    n_shot=50
    ExpNum = 1000
    train_model = False; FewShotEvaluation = True; 
    save_model = False; load_model = True;
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Train_X,Train_Y,Test_Y,Test_Y,SupportSet_X,SupportSet_Y = LoadDataFMnist(labels_out = labels_out)
    Train_X_triplets, Train_Y_triplets, Train_Index_triplets = CreateTriplets(Train_X,Train_Y,TripletSetSize=TripletSetSize)
    Test_X_triplets, Test_Y_triplets, Test_Index_triplets = CreateTriplets(Train_X,Train_Y,TripletSetSize=TripletTestSize)
    model = TripletNetModel(device)
    # Optimizer - Adam, Learning rate scheduler - Exponential learning rate
    optimizer = Adam(model.parameters(),lr = lr)
    scheduler = ExponentialLR(optimizer, gamma=gamma)
    # Train model
    if train_model:
        losses = []; test_losses = []; dists_plus = []; dists_minus = []
        plt.figure()
        cur_loss = np.inf
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
                dists_plus_temp_temp = dist_plus.detach().cpu().numpy()
                dists_minus_temp_temp = dist_minus.detach().cpu().numpy()
                norm = np.exp(dists_plus_temp_temp) + np.exp(dists_minus_temp_temp)
                dists_plus_temp =+ np.mean(np.exp(dists_plus_temp_temp)/norm)
                dists_minus_temp =+ np.mean(np.exp(dists_minus_temp_temp)/norm)
                loss_cur.append(loss.item())
            scheduler.step()
            losses.append(np.mean(loss_cur))
            test_losses.append(evaluate_test(Test_X_triplets,TripletTestSize,batch_size))
            dists_plus.append(dists_plus_temp); dists_minus.append(dists_minus_temp); 
            print('Train Loss: ' + str(losses[-1]))
            print('Test Loss: ' + str(test_losses[-1]))
            print('Similarity: ' + str(dists_plus[-1]))
            print('Disimilarity: ' + str(dists_minus[-1]))
            cur_loss = min(test_losses[-1],cur_loss)
            if save_model and cur_loss == test_losses[-1]:
                torch.save(model.state_dict(), '../models/FewShotLearningTripletNetwork/TripletModel')
            plt.subplot(2,2,1)
            plt.semilogy(np.linspace(1,epoch+1,epoch+1),losses)
            plt.xlabel('Iteartion [#]')
            plt.ylabel('Loss')
            plt.title('Training loss')
            plt.subplot(2,2,2)
            plt.semilogy(np.linspace(1,epoch+1,epoch+1),test_losses)
            plt.xlabel('Iteartion [#]')
            plt.ylabel('Loss')
            plt.title('Test loss')
            plt.subplot(2,2,3)
            plt.plot(np.linspace(1,epoch+1,epoch+1),dists_plus)
            plt.xlabel('Iteartion [#]')
            plt.ylabel('Probability')
            plt.title('Similarity')
            plt.subplot(2,2,4)
            plt.plot(np.linspace(1,epoch+1,epoch+1),dists_minus)
            plt.xlabel('Iteartion [#]')
            plt.ylabel('Probability')
            plt.title('Dissimilarity')
            plt.suptitle('Iteration number: ' + str(epoch + 1))
            plt.show()
            plt.pause(0.02)
    if load_model:
        model = LoadBestModel()
    # Evaluate few shot learning scenario
    if FewShotEvaluation:
        k_way_temp = 3
        accN,stdN = CalculateNshotsAccuray(n_shot,ExpNum,SupportSet_X,SupportSet_Y,labels_out,k_way_temp,device)
        n_shot_temp = 5
        accK,stdK = CalculateKWaysAccuray(n_shot_temp,k_way,ExpNum,SupportSet_X,SupportSet_Y,labels_out,device)
        savgolFilterOn = True
        plt.figure()
        plt.subplot(1,2,1)
        if savgolFilterOn:
            plt.plot(np.linspace(1,n_shot,n_shot),savgol_filter(accN,7,1))
        else:
            plt.plot(np.linspace(1,n_shot,n_shot),accN)
        plt.xlabel('N-shots [#]')
        plt.ylabel('Accuracy [%]')
        plt.title('Accuracy as a function of the number of shots, $k_{ways}$ = ' + str(k_way_temp))
        plt.subplot(1,2,2)
        plt.plot(np.linspace(1,len(accK),len(accK)),accK)
        plt.xlabel('K-ways [#]')
        plt.ylabel('Accuracy [%]')
        plt.title('Accuracy as a function of k number of classes in support set, $n_{shots}$ = ' + str(n_shot_temp))
        plt.suptitle('Results averaged over ' + str(ExpNum) + ' experiments')
        
    
