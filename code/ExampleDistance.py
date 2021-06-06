# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 19:04:51 2021

@author: danie
"""

# %% Imports
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from DataBase import FashionMNIST_t
from TrainTripletNetwork import LoadBestModel
from Losses import tripletLoss, CustomLoss


# %% Convert tensor to image

def convert_to_imshow_format(image):
    image=image/2+0.5
    image=image.detach().cpu().numpy()
    return image.transpose(1,2,0)

# %% Load data
batch_size = 4
trainset = FashionMNIST_t('../data', train=True, download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914,),(0.2023,))]))
testset = FashionMNIST_t('../data', train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914,),(0.2023,))]))
train_loader = torch.utils.data.DataLoader(trainset,batch_size = batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset,batch_size = batch_size ,shuffle=True)
model = LoadBestModel(load_model=True)

# %% Show images

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataiter= iter(train_loader)
anchor, negative, positive = dataiter.next()
anchor = anchor.to(device).requires_grad_();
negative = negative.to(device).requires_grad_();
positive = positive.to(device).requires_grad_();
dist_plus, dist_minus, embedded_anchor, embedded_positive, embedded_negative = model(anchor, positive, negative)
loss1 = tripletLoss(embedded_anchor,embedded_positive,embedded_negative).detach().cpu().numpy()
loss2 = CustomLoss(dist_plus,dist_minus).detach().cpu().numpy()
dist_plus = dist_plus.detach().cpu().numpy()
dist_minus = dist_minus.detach().cpu().numpy()
embedded_anchor = embedded_anchor.detach().cpu().numpy()
embedded_positive = embedded_positive.detach().cpu().numpy()
embedded_negative = embedded_negative.detach().cpu().numpy()

plt.figure()
for idx,image in enumerate(anchor):
    plt.subplot(3,4,idx+1)
    plt.imshow(convert_to_imshow_format(image)[:,:,0],cmap=plt.cm.binary)
    plt.title('Anchor')
for idx,image in enumerate(positive):
    plt.subplot(3,4,idx+1 + batch_size)
    plt.imshow(convert_to_imshow_format(image)[:,:,0],cmap=plt.cm.binary)
    plt.title('Positive')
for idx,image in enumerate(negative):
    plt.subplot(3,4,idx+1 + batch_size * 2)
    plt.imshow(convert_to_imshow_format(image)[:,:,0],cmap=plt.cm.binary)
    plt.title('Negative')

