# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 15:08:36 2021

@author: danie
"""

# %% Imports

import torch
import torch.nn as nn

# %% Triplet loss

def tripletLoss(margin = 0.2, p=2):
    triplet_loss = nn.TripletMarginLoss(margin=margin, p=p)
    return triplet_loss

# %% Main

if __name__ == '__main__':
    triplet_loss = tripletLoss()
    anchor = torch.randn(100, 128, requires_grad=True)
    positive = torch.randn(100, 128, requires_grad=True)
    negative = torch.randn(100, 128, requires_grad=True)
    output = triplet_loss(anchor, positive, negative)
    output.backward()
