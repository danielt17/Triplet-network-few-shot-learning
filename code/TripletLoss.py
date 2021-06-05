# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 15:08:36 2021

@author: danie
"""

# %% Imports

import torch
import torch.nn as nn

# %% Triplet loss

def tripletLoss(anchor,positive,negative,margin = 0.2, p=2):
    triplet_loss = nn.TripletMarginLoss(margin=margin, p=p)
    loss = triplet_loss(anchor, positive, negative)
    return loss

def CustomLoss(dist_plus,dist_minus,margin = 1):
    norm = torch.add(torch.exp(dist_plus),torch.exp(dist_minus))
    dplus = torch.div(torch.exp(dist_plus),norm)
    dminus = torch.div(torch.exp(dist_minus),norm)
    loss = torch.mean(torch.add(torch.square(dplus),torch.square(torch.sub(dminus,margin))))
    return loss

# %% Main

if __name__ == '__main__':
    # example 1
    anchor = torch.randn(100, 128, requires_grad=True)
    positive = torch.randn(100, 128, requires_grad=True)
    negative = torch.randn(100, 128, requires_grad=True)
    output = tripletLoss(anchor, positive, negative)
    output.backward()
    # example 2
    dist_plus = torch.randn(100, 128, requires_grad=True)
    dist_minus = torch.randn(100, 128, requires_grad=True)
    output = CustomLoss(dist_plus,dist_minus)
    output.backward()
    
    
