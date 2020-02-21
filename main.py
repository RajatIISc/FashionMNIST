# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 17:05:43 2020

@author: Rajat
"""

"""
Code to use the saved models for testing
"""

import numpy as np
import pdb
import os
from tqdm import tqdm

from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.datasets import FashionMNIST
from torchvision import transforms

from utils import AverageMeter


def test(model, testloader, flag):
    """ Training the model using the given dataloader for 1 epoch.
    Input: Model, Dataset, optimizer,
    """

    model.eval()
    avg_loss = AverageMeter("average-loss")

    y_gt = []
    y_pred_label = []

    for batch_idx, (img, y_true) in enumerate(testloader):
        img = Variable(img)
        y_true = Variable(y_true)
        if(flag==0):
            img = img.reshape(-1, 28*28)
        out = model(img)
        y_pred = F.softmax(out, dim=1)
        y_pred_label_tmp = torch.argmax(y_pred, dim=1)

        loss = F.cross_entropy(out, y_true)
        avg_loss.update(loss, img.shape[0])

        # Add the labels
        y_gt += list(y_true.numpy())
        y_pred_label += list(y_pred_label_tmp.numpy())

    return avg_loss.avg, y_gt, y_pred_label


# Definition of the network
class FFNN_class(nn.Module):
    def __init__(self,num_class):
        super().__init__()
        self.FC1 = nn.Linear(784,500)
        self.relu = nn.ReLU()
        self.FC2 = nn.Linear(500,num_class)
        
    def forward(self,x):
        out = self.FC1(x)
        out = self.relu(out)
        out = self.FC2(out)
        return out
    
class My_CNN(nn.Module):
    def __init__(self,num_class):
        super().__init__()
        self.Conv1 = nn.Conv2d(in_channels = 1, out_channels = 6,kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Conv2 = nn.Conv2d(in_channels = 6, out_channels = 12, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.FC1 = nn.Linear(12*4*4,120)
        self.relu3 = nn.ReLU()
        self.FC2 = nn.Linear(120,60)
        self.relu4 = nn.ReLU()
        self.FC3 = nn.Linear(60,num_class)
        
        #forward pass
    def forward(self,x):
        out = self.Conv1(x)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.Conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        #print(out.shape)
        out = out.reshape(-1, 12*4*4)
        #print(out.shape)
        out = self.FC1(out)
        out = self.relu3(out)
        out = self.FC2(out)
        out = self.relu4(out)
        out = self.FC3(out)
            # softmax is not required as we are using crossentropy
        return out

if __name__ == "__main__":

    trans_img = transforms.Compose([transforms.ToTensor()])
    dataset = FashionMNIST("./data/", train=False, transform=trans_img, download=True)
    testloader = DataLoader(dataset, batch_size=1024, shuffle=False)

    model_MLP = FFNN_class(10)
    model_MLP.load_state_dict(torch.load("./models/model.ckpt"))

    model_conv_net = My_CNN(10)
    model_conv_net.load_state_dict(torch.load("./models/model_cnn.ckpt"))

    loss, gt, pred = test(model_MLP, testloader, 0)
    with open("multi-layer-net.txt", 'w') as f:
        f.write("Loss on Test Data : {}\n".format(loss))
        f.write("Accuracy on Test Data : {}\n".format(np.mean(np.array(gt) == np.array(pred))))
        f.write("gt_label,pred_label \n")
        for idx in range(len(gt)):
            f.write("{},{}\n".format(gt[idx], pred[idx]))

    loss, gt, pred = test(model_conv_net, testloader, 1)
    with open("convolution-neural-net.txt", 'w') as f:
        f.write("Loss on Test Data : {}\n".format(loss))
        f.write("Accuracy on Test Data : {}\n".format(np.mean(np.array(gt) == np.array(pred))))
        f.write("gt_label,pred_label \n")
        for idx in range(len(gt)):
            f.write("{},{}\n".format(gt[idx], pred[idx]))