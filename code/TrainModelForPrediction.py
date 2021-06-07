# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 19:05:37 2021

@author: danie
"""

# %% Imports

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import numpy as np
from barbar import Bar
import matplotlib.pyplot as plt

from VisualizeTSNE import LoadData, ModelPrediction
from TrainTripletNetwork import LoadBestModel
from PredictionNetworks import FashionCNNmodel


# %% Create data set

def TripletFeaturesDataSet(data_triplet_train,target_train,data_triplet_test,target_test,batch_size):
    x_train = list(data_triplet_train); y_train = list(target_train)
    x_test = list(data_triplet_test); y_test = list(target_test)
    x_train = torch.Tensor(x_train); y_train = torch.Tensor(y_train)
    x_test = torch.Tensor(x_test); y_test = torch.Tensor(y_test)
    trainTripletFeaturesSet = TensorDataset(x_train,y_train)
    testTripletFeaturesSet = TensorDataset(x_test,y_test)
    trainTripletFeaturesLoader = DataLoader(trainTripletFeaturesSet,batch_size = batch_size)
    testTripletFeaturesLoader = DataLoader(testTripletFeaturesSet,batch_size = batch_size)
    return trainTripletFeaturesLoader,testTripletFeaturesLoader

def GetStandardAndTripletFeaturesDataSets(device,batch_size):
    modelTriplet = LoadBestModel(True,1)
    trainset ,testset,data_train,data_test,target_train,target_test,classes_dict = LoadData(device,batch_size = 1)
    data_triplet_train = ModelPrediction(trainset,modelTriplet,device)
    data_triplet_test = ModelPrediction(testset,modelTriplet,device)
    trainTripletFeaturesLoader,testTripletFeaturesLoader = TripletFeaturesDataSet(data_triplet_train,target_train,data_triplet_test,target_test,batch_size)
    trainset ,testset,_,_,_,_,_ = LoadData(device,batch_size = batch_size)
    return trainset,testset,data_train,data_test,target_train,target_test,classes_dict,data_triplet_train,data_triplet_test,trainTripletFeaturesLoader,testTripletFeaturesLoader

def evaluate_prediction(test,error):
    model.eval()
    losses = []
    accurcay = []
    for batch_idx, data in enumerate(Bar(test)):
        batch = data[0].to(device)
        label = data[1].to(device)
        outputs = model(batch)
        loss = error(outputs, label)
        losses.append(loss.item())
        accurcay.append(CalcualteAccuracy(outputs,label))
    losses = np.mean(losses)
    accurcay = np.mean(accurcay)
    return losses,accurcay

def CalcualteAccuracy(outputs,label):
    pred = np.argmax(outputs.detach().cpu().numpy(),axis =1)
    labels = label.detach().cpu().numpy()
    accuracy = 100 * np.sum(pred==labels)/len(pred)
    return accuracy

# %% Main

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    epochs = 5
    error = nn.CrossEntropyLoss()
    lr = 1e-3
    trainset,testset,data_train,data_test,target_train,target_test,classes_dict,data_triplet_train,data_triplet_test,trainTripletFeaturesLoader,testTripletFeaturesLoader = GetStandardAndTripletFeaturesDataSets(device,batch_size)
    model = FashionCNNmodel(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []; losses_test = [];
    accuracy = []; accuracy_test = [];
    plt.figure()
    for epoch in range(epochs):
        print('Epoch number:' + str(epoch +1))
        loss_cur = []; accuracy_cur = [];
        for batch_idx, data in enumerate(Bar(trainset)):
            batch = data[0].to(device).requires_grad_()
            label = data[1].to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = error(outputs, label)
            loss.backward()
            optimizer.step()
            loss_cur.append(loss.item())
            accuracy_cur.append(CalcualteAccuracy(outputs,label))
        losses.append(np.mean(loss_cur))
        accuracy.append(np.mean(accuracy_cur))
        losses_test_cur, accurcay_test_cur = evaluate_prediction(testset,error)
        losses_test.append(losses_test_cur)
        accuracy_test.append(accurcay_test_cur)
        print('Training loss: ' + str(losses[-1]) + ' Test loss: ' + str(losses_test[-1]))
        print('Training accuracy: ' + str(losses[-1]) + ' Test accuracy: ' + str(losses_test[-1]))
        plt.clf()
        plt.subplot(2,2,1)
        plt.semilogy(losses)
        plt.xlabel('Iteration [#]')
        plt.ylabel('Loss')
        plt.title('Training loss')
        plt.subplot(2,2,2)
        plt.plot(accuracy)
        plt.xlabel('Iteration [#]')
        plt.ylabel('Accuracy')
        plt.title('Training accuracy')
        plt.subplot(2,2,3)
        plt.semilogy(losses_test)
        plt.xlabel('Iteration [#]')
        plt.ylabel('Loss')
        plt.title('Test loss')
        plt.subplot(2,2,4)
        plt.plot(accuracy_test)
        plt.xlabel('Iteration [#]')
        plt.ylabel('Accuracy')
        plt.title('Test accuracy')
        plt.suptitle('Epoch number: ' + str(epoch +1))
        plt.show()
        plt.pause(0.02)
            