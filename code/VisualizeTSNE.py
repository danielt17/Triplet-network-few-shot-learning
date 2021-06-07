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
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE 

# %% Functions

def LoadData(device):
    data = torchvision.datasets.FashionMNIST('../FashionMnist',download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914,),(0.2023,))]))
    train, test = torch.utils.data.random_split(data,[50000,10000])
    trainset = torch.utils.data.DataLoader(train,batch_size=1,shuffle=False)
    ind_train = trainset.dataset.indices
    testset = torch.utils.data.DataLoader(test,batch_size=1,shuffle=False)
    ind_test = testset.dataset.indices
    data_train = data.data.numpy()[ind_train]; data_train = np.reshape(data_train,(data_train.shape[0],28*28))
    data_test = data.data.numpy()[ind_test]; data_test = np.reshape(data_test,(data_test.shape[0],28*28))
    target_train = data.targets.numpy()[ind_train]
    target_test = data.targets.numpy()[ind_test]
    classes_dict = data.class_to_idx
    return trainset ,testset,data_train,data_test,target_train,target_test,classes_dict             

def ModelPrediction(dataloader,model):
    ls = []
    for batch_idx, data in enumerate(dataloader):
        data_in = data[0]
        data_in = data_in.to(device)
        _,_,output,_,_ = model(data_in,data_in,data_in)
        temp_output = output.detach().cpu().numpy()
        temp_output = np.reshape(temp_output,(-1,))
        ls.append(temp_output)
    return np.array(ls)

def PresentTSNE(dataset_numpy,targetset_numpy,classes_dict,title):
    data_test_tsne = TSNE(n_components=2).fit_transform(dataset_numpy)
    plt.figure()
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
    target_ids = range(len(list(classes_dict.values())))
    for i,c, label in zip(target_ids, colors, list(classes_dict.keys())):
        plt.scatter(data_test_tsne[targetset_numpy==i,0],data_test_tsne[targetset_numpy==i,1],c=c,label = label)
    plt.legend()
    plt.xlabel('First component')
    plt.ylabel('Second component')
    plt.title(title)
    plt.show()
    
# %% Main

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LoadBestModel(True,1)
    trainset ,testset,data_train,data_test,target_train,target_test,classes_dict = LoadData(device)
    data_model_test = ModelPrediction(testset,model)
    PresentTSNE(data_test,target_test,classes_dict,title='TSNE over standard dataset')
    PresentTSNE(data_model_test,target_test,classes_dict,title='TSNE over model dataset')
    
    
    