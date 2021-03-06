# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 12:15:25 2021

@author: danie
"""

# %% Imports

import torch
import torchvision
from torchvision import transforms
from TrainTripletNetwork import LoadBestModel
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.manifold import TSNE 
from barbar import Bar

# %% Setup plot

font = {'family' : 'normal',
        'size'   : 14}

matplotlib.rc('font', **font)

# %% Functions

def LoadData(device,batch_size=1):
    '''
    Description:
        Load Fashion-MNIST data and turn them to numpy objects
    Inputs:
        device: cpu or cuda enbaled gpu
        batch_size: load data one by one for plotting
    Returns: 
        trainset: train dataloader
        testset: test dataloader
        data_train: train numpy array
        data_test: test numpy array
        target_train: train labels numpy array
        target_test: test labels numpy array
        classes_dict: dictionary of label - class
    '''
    data = torchvision.datasets.FashionMNIST('../FashionMnist',download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914,),(0.2023,))]))
    train, test = torch.utils.data.random_split(data,[50000,10000])
    trainset = torch.utils.data.DataLoader(train,batch_size=batch_size,shuffle=False)
    ind_train = trainset.dataset.indices
    testset = torch.utils.data.DataLoader(test,batch_size=batch_size,shuffle=False)
    ind_test = testset.dataset.indices
    data_train = data.data.numpy()[ind_train]; data_train = np.reshape(data_train,(data_train.shape[0],28*28))
    data_test = data.data.numpy()[ind_test]; data_test = np.reshape(data_test,(data_test.shape[0],28*28))
    target_train = data.targets.numpy()[ind_train]
    target_test = data.targets.numpy()[ind_test]
    classes_dict = data.class_to_idx
    return trainset ,testset,data_train,data_test,target_train,target_test,classes_dict             

def ModelPrediction(dataloader,model,device):
    '''
    Description:
        Produce model outputs with respect to a given dataloader
    Inputs:
        dataloader: dataloader object
        model: triplet model
        device: cpu or cuda enbaled gpu
    Returns: 
        ls: numpy array of vector embedding output
    '''
    ls = []
    for batch_idx, data in enumerate(Bar(dataloader)):
        data_in = data[0]
        data_in = data_in.to(device)
        _,_,output,_,_ = model(data_in,data_in,data_in)
        temp_output = output.detach().cpu().numpy()
        temp_output = np.reshape(temp_output,(-1,))
        ls.append(temp_output)
    return np.array(ls)

def PresentTSNE(dataset_numpy,targetset_numpy,classes_dict):
    '''
    Description:
        This function presents the T-SNE visualization with respect to a given dataset examples and labels
    Inputs:
        dataset_numpy: dataset numpy array
        targetset_numpy: target labels numpy array
        classes_dict: dictionary of label - class
        title: title to plot
    Returns: 
        2D and 3D plots of TSNE
    '''
    data_test_tsne = TSNE(n_components=2).fit_transform(dataset_numpy)
    data_test_tsne_3d = TSNE(n_components=3).fit_transform(dataset_numpy)
    plt.figure()
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
    target_ids = range(len(list(classes_dict.values())))
    for i,c, label in zip(target_ids, colors, list(classes_dict.keys())):
        plt.scatter(data_test_tsne[targetset_numpy==i,0],data_test_tsne[targetset_numpy==i,1],c=c,label = label)
    plt.legend()
    plt.xlabel('First component')
    plt.ylabel('Second component')
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i,c, label in zip(target_ids, colors, list(classes_dict.keys())):
        ax.scatter(data_test_tsne_3d[targetset_numpy==i,0],data_test_tsne_3d[targetset_numpy==i,1],data_test_tsne_3d[targetset_numpy==i,2],c=c,label = label)
    plt.legend()
    ax.set_xlabel('First component')
    ax.set_ylabel('Second component')
    ax.set_zlabel('Third component')
    plt.show()
    
# %% Main

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LoadBestModel(True,1)
    trainset ,testset,data_train,data_test,target_train,target_test,classes_dict = LoadData(device)
    data_model_test = ModelPrediction(testset,model,device)
    PresentTSNE(data_test,target_test,classes_dict) # title='TSNE over standard dataset'
    PresentTSNE(data_model_test,target_test,classes_dict) # title='TSNE over model dataset'
    
    
    