# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 19:04:39 2021

@author: danie
"""

# %% Imports

import torch
import torchvision
from torchvision import transforms
import numpy as np 

# %% Functions

def LoadDataFMnist(labels_out = [7,8,9]):
    '''
    Description:
        Splits data into train set and support set
    Inputs:
        labels_out: labels in support set
    Returns: 
        Train_X: Train set inputs
        Train_Y: Train set ouputs
        Test_X: Test set inputs
        Test_Y: Test set ouputs
        SupportSet_X: Support set inputs
        SupportSet_Y: Support set ouputs
    '''
    data = torchvision.datasets.FashionMNIST('../FashionMnist',download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914,),(0.2023,))]))
    X_full = data.data.numpy()
    labels = data.targets.numpy()
    indForTrainLs = []
    for label in labels_out:
        indForTrainLs.append((labels!=label))
    indForTrain = np.ones((len(labels),),dtype = bool)
    for ls in indForTrainLs:
        indForTrain = indForTrain*ls
    TrainTestSets_X = X_full[indForTrain]; TrainTestSets_Y = labels[indForTrain]
    SupportSet_X = X_full[~indForTrain]; SupportSet_Y = labels[~indForTrain]
    split_size = np.int64(len(TrainTestSets_X)*0.8)
    Train_X = TrainTestSets_X[:split_size]; Train_Y =  TrainTestSets_Y[:split_size]
    Test_X = TrainTestSets_X[split_size:]; Test_Y =  TrainTestSets_Y[split_size:]
    return Train_X,Train_Y,Test_X,Test_Y,np.reshape(SupportSet_X,(SupportSet_X.shape[0],1,28,28)),SupportSet_Y

def CreateTriplets(X,Y,TripletSetSize=60000):
    '''
    Description:
        Creates triplet PyTorch tensors
    Inputs:
        X: Input set
        Y: Output set
        TripletSetSize: output triplet sets size
    Returns: 
        X_triplets: Inputs of triplet set
        Y_triplets: Labels of triplet set
        Index_triplets: Index of triplet images
    '''
    Y_triplets = []; Index_triplets = [];
    anchor_array = np.zeros((TripletSetSize,1,28,28))
    positive_array = np.zeros((TripletSetSize,1,28,28))
    negative_array = np.zeros((TripletSetSize,1,28,28))
    labels = np.unique(Y)
    for ind in range(TripletSetSize):
        anchor_label = np.random.choice(labels)
        negative_label = np.random.choice(labels[labels!=anchor_label])
        positives = np.where(Y==anchor_label)[0]
        negatives = np.where(Y==negative_label)[0]
        anchor_ind =  np.random.choice(positives)
        positive_ind = np.random.choice(positives[positives!=anchor_ind])
        negative_ind = np.random.choice(negatives)
        anchor_array[ind] = X[anchor_ind:anchor_ind+1]
        positive_array[ind] = X[positive_ind:positive_ind+1]
        negative_array[ind] = X[negative_ind:negative_ind+1]
        Y_triplets.append((anchor_label,anchor_label,negative_label))
        Index_triplets.append((anchor_ind,positive_ind,negative_ind))
    X_triplets = [torch.from_numpy(np.float32(anchor_array)),torch.from_numpy(np.float32(positive_array)),torch.from_numpy(np.float32(negative_array))]
    return X_triplets, Y_triplets, Index_triplets 
    
    
def SupportSetAndQuery(SupportSet_X,SupportSet_Y,labels_out,k_way=2,n_shot=3):
    '''
    Description:
        Creates support set and query with respect to given parameters 
    Inputs:
        SupportSet_X: Support set inputs
        SupportSet_Y: Support set outputs
        labels_out: labels in support set
        k_way: number of classes in actual support set
        n_shot: number of shots in each class in support set
    Returns: 
        Query: Query image
        QueryLabel: Query ground truth label
        classes: classes in actual support et
        SupportSet: actual support set of size k*n
    '''
    labels_out = np.asarray(labels_out)
    QueryLabel = np.random.choice(labels_out)
    QueryInd = np.random.choice(np.where(SupportSet_Y==QueryLabel)[0])
    Query = SupportSet_X[QueryInd:QueryInd+1]
    Query = np.repeat(Query,n_shot,axis = 0)
    classes = []
    otherLabels = np.where(labels_out!=QueryLabel)[0]
    while len(classes) < (k_way - 1):
        labelNew = np.random.choice(labels_out[otherLabels])
        if labelNew not in classes:
            classes.append(labelNew)
    classes.append(QueryLabel)
    SupportSet = np.zeros((len(classes),n_shot,1,28,28))
    for ind,label in enumerate(classes):
        cur_label_examples = np.where(SupportSet_Y==label)[0]
        inds_cur_label = np.random.choice(cur_label_examples, size=n_shot, replace=False)
        SupportSet[ind] = SupportSet_X[inds_cur_label]
    SupportSet = list(SupportSet)
    for i in range(len(SupportSet)):
        SupportSet[i] = torch.from_numpy(np.float32(SupportSet[i]))
    return torch.from_numpy(np.float32(Query)),QueryLabel,classes,SupportSet

# %% Main

if __name__ == '__main__':
    labels_out = [7,8,9]
    TripletSetSize = 60000
    TripletTestSize = np.int64(60000*0.2)
    k_way=2
    n_shot=50
    Train_X,Train_Y,Test_X,Test_Y,SupportSet_X,SupportSet_Y = LoadDataFMnist(labels_out = labels_out)
    Train_X_triplets, Train_Y_triplets, Train_Index_triplets = CreateTriplets(Train_X,Train_Y,TripletSetSize=TripletSetSize)
    Test_X_triplets, Test_Y_triplets, Test_Index_triplets = CreateTriplets(Train_X,Train_Y,TripletSetSize=TripletTestSize)
    Query,QueryLabel,classes,SupportSet = SupportSetAndQuery(SupportSet_X,SupportSet_Y,labels_out,k_way=k_way,n_shot=n_shot)
    
    
    
    
    
    
    
    
    
    
    
    