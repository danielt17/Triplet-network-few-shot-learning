# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 19:05:37 2021

@author: danie
"""

# %% Imports

import torch
import torch.nn as nn
import numpy as np
from barbar import Bar
import matplotlib
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from VisualizeTSNE import LoadData, ModelPrediction
from TrainTripletNetwork import LoadBestModel
from PredictionNetworks import FashionCNNmodel

# %% Setup plot

font = {'family' : 'normal',
        'size'   : 14}

matplotlib.rc('font', **font)

# %% Create data set

def GetStandardAndTripletFeaturesDataSets(device,batch_size):
    '''
    Inputs:
        data_triplet_train: cpu or cuda enbaled gpu
        batch_size: required batch size
    Returns: 
        trainset: train dataloader
        testset: test dataloader
        data_train: train numpy array
        data_test: test numpy array
        target_train: train labels numpy array
        target_test: test labels numpy array
        classes_dict: dictionary of label - class
        data_triplet_train: train, triplet feature vector numpy array
        data_triplet_test: test, triplet feature vector numpy array
    '''
    modelTriplet = LoadBestModel(True,1)
    trainset ,testset,data_train,data_test,target_train,target_test,classes_dict = LoadData(device,batch_size = 1)
    data_triplet_train = ModelPrediction(trainset,modelTriplet,device)
    data_triplet_test = ModelPrediction(testset,modelTriplet,device)
    trainset ,testset,_,_,_,_,_ = LoadData(device,batch_size = batch_size)
    return trainset,testset,data_train,data_test,target_train,target_test,classes_dict,data_triplet_train,data_triplet_test

def evaluate_prediction(test,error):
    '''
    Inputs:
        test: test dataloader
        error: loss object
    Returns: 
        losses: loss current epoch on test set
        accurcay: accuracy current epoch on test set
    '''
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
    '''
    Inputs:
        outputs: CNN output probabilites
        label: ground truth label
    Returns: 
        accuracy: accuracy
    '''
    pred = np.argmax(outputs.detach().cpu().numpy(),axis =1)
    labels = label.detach().cpu().numpy()
    accuracy = 100 * np.sum(pred==labels)/len(pred)
    return accuracy

# %% Main

if __name__ == '__main__':
    # Hyperparameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    epochs = 50
    error = nn.CrossEntropyLoss()
    lr = 1e-5
    Train_Deep_model = False
    save_model = False;
    load_model = True;
    trainset,testset,data_train,data_test,target_train,target_test,classes_dict,data_triplet_train,data_triplet_test = GetStandardAndTripletFeaturesDataSets(device,batch_size)
    # Train model
    if Train_Deep_model:
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
            print('Training accuracy: ' + str(accuracy[-1]) + ' Test accuracy: ' + str(accuracy_test[-1]))
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
        if save_model:
            torch.save(model.state_dict(), '../models/modelStandardFashionMnist/modelFashionMnist')
    if load_model:
        model = FashionCNNmodel(device)
        model.load_state_dict(torch.load('../models/modelStandardFashionMnist/modelFashionMnist'))
    # Evaluate classical ML classifiers
    print('Prediction: ')
    acc_model = evaluate_prediction(testset,error)[1]
    svm = SVC(kernel='linear')
    svm.fit(data_triplet_train,target_train)
    acc_svm = svm.score(data_triplet_test,target_test) * 100
    KNN = KNeighborsClassifier(n_neighbors=100)
    KNN.fit(data_triplet_train,target_train)
    acc_knn = KNN.score(data_triplet_test,target_test) * 100
    print('DNN model accuracy: ' + str(acc_model)[:5] + '%')
    print('Linear SVM accuracy: ' + str(acc_svm)[:5] + '%')
    print('KNN accuracy: ' + str(acc_knn)[:5] + '%')
    
    
            