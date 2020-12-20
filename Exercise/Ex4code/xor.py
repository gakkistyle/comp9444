################################################################
# xor.py
# UNSW, COMP9444
# Alan Blair
#
# Pytorch program to train a 2-layer NN on the XOR task
#
# a. Run the code ten times by typing
#     python3 xor.py
#
#    For how many runs does it reach the global minimum?
#    For how many runs does it reach a local minimum?
# 
# b. Keeping the learning rate fixed at 0.1, can you find values of
#    momentum and initial weight size for which the code converges
#    relatively quickly to the global minimum on every run?

import torch
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr',  type=float, default=0.1, help='learning rate')
parser.add_argument('--mom', type=float, default=0.0, help='momentum')
parser.add_argument('--init',type=float, default=1.0, help='initial weight size')

args = parser.parse_args()

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # define structure of the network here
        self.in_hid  = torch.nn.Linear(2,2)
        self.hid_out = torch.nn.Linear(2,1)
    def forward(self, input):
        # apply network and return output
        hid_sum = self.in_hid(input)
        hidden  = torch.tanh(hid_sum)
        out_sum = self.hid_out(hidden)
        output  = torch.sigmoid(out_sum)
        return(output)

device = 'cpu'

input  = torch.Tensor([[0,0],[0,1],[1,0],[1,1]])
target = torch.Tensor([[0],[1],[1],[0]])

xor_dataset  = torch.utils.data.TensorDataset(input,target)
train_loader = torch.utils.data.DataLoader(xor_dataset,batch_size=4)

# create neural network according to model specification
net = MyModel().to(device) # CPU or GPU

# initialize weight values
net.in_hid.weight.data.normal_(0,args.init)
net.hid_out.weight.data.normal_(0,args.init)

# choose between SGD, Adam or other optimizer
optimizer = torch.optim.SGD(net.parameters(),lr=args.lr,momentum=args.mom)

epochs = 10000

for epoch in range(1, epochs):
    #train(net, device, train_loader, optimizer)
    for batch_id, (data,target) in enumerate(train_loader):
        optimizer.zero_grad() # zero the gradients
        output = net(data)    # apply network
        loss = F.binary_cross_entropy(output,target)
        loss.backward()       # compute gradients
        optimizer.step()      # update weights
        if epoch % 10 == 0:
            print('ep%3d: loss = %7.4f' % (epoch, loss.item()))
        if loss < 0.01:
            exit()
