# kuzu.py
# COMP9444, CSE, UNSW

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        # INSERT CODE HERE
        self.linear = nn.Linear(28*28, 10)
        self.log_soft_max = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x_1 = x.view(x.size(0),-1)
        x_2 = self.log_soft_max(self.linear(x_1))
        return x_2 # CHANGE CODE HERE

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        # INSERT CODE HERE
        self.hid = nn.Linear(28*28,200)
        self.tanh = nn.Tanh()
        self.out = nn.Linear(200,10)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x_1 = x.view(x.size(0),-1)
        x_2 = self.hid(x_1)
        x_3 = self.tanh(x_2)
        x_4 = self.out(x_3)
        x_5 = self.log_softmax(x_4)
        return x_5 # CHANGE CODE HERE

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        # INSERT CODE HERE
        self.conv_1 = nn.Conv2d(1,32,6,padding=2)
        self.conv_2 = nn.Conv2d(32,64,6,padding=2)
        self.linear_1 = nn.Linear(43264,64) 
        self.liear_2 = nn.Linear(64,10)
        self.re_lu = nn.ReLU()
        self.log_soft_max = nn.LogSoftmax()

    def forward(self, x):
        x_1 = self.re_lu(self.conv_2(self.re_lu(self.conv_1(x))))
        h = x_1.view(x_1.size(0),-1)
        output = self.log_soft_max(self.liear_2(self.re_lu(self.linear_1(h))))
        return output
    