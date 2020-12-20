# spiral_main.py
# COMP9444, CSE, UNSW

import torch
import torch.utils.data
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from spiral import PolarNet, RawNet, graph_hidden

def train(net, train_loader, optimizer):
    total=0
    correct=0
    for batch_id, (data,target) in enumerate(train_loader):
        optimizer.zero_grad()    # zero the gradients
        output = net(data)       # apply network
        loss = F.binary_cross_entropy(output,target)
        loss.backward()          # compute gradients
        optimizer.step()         # update weights
        pred = (output >= 0.5).float()
        correct += (pred == target).float().sum()
        total += target.size()[0]
        accuracy = 100*correct/total

    if epoch % 100 == 0:
        print('ep:%5d loss: %6.4f acc: %5.2f' %
             (epoch,loss.item(),accuracy))

    return accuracy

def graph_output(net):
    xrange = torch.arange(start=-7,end=7.1,step=0.01,dtype=torch.float32)
    yrange = torch.arange(start=-6.6,end=6.7,step=0.01,dtype=torch.float32)
    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    grid = torch.cat((xcoord.unsqueeze(1),ycoord.unsqueeze(1)),1)

    with torch.no_grad(): # suppress updating of gradients
        net.eval()        # toggle batch norm, dropout
        output = net(grid)
        net.train() # toggle batchnorm, dropout back again

        pred = (output >= 0.5).float()

        # plot function computed by model
        plt.clf()
        plt.pcolormesh(xrange,yrange,pred.cpu().view(yrange.size()[0],xrange.size()[0]), cmap='Wistia')

# command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--net',type=str,default='raw',help='polar or raw')
parser.add_argument('--init',type=float,default=0.1,help='initial weight size')
parser.add_argument('--hid',type=int,default='10',help='number of hidden units')
parser.add_argument('--lr',type=float,default=0.01,help='learning rate')
parser.add_argument('--epochs',type=int,default='100000',help='max training epochs')
args = parser.parse_args()

df = pd.read_csv('spirals.csv')

data = torch.tensor(df.values,dtype=torch.float32)

num_input = data.shape[1] - 1

full_input  = data[:,0:num_input]
full_target = data[:,num_input:num_input+1]

train_dataset = torch.utils.data.TensorDataset(full_input,full_target)
train_loader  = torch.utils.data.DataLoader(train_dataset,batch_size=97)

# choose network architecture
if args.net == 'polar':
    net = PolarNet(args.hid)
else:
    net = RawNet(args.hid)

if list(net.parameters()):
    # initialize weight values
    for m in list(net.parameters()):
        m.data.normal_(0,args.init)

    # use Adam optimizer
    optimizer = torch.optim.Adam(net.parameters(),eps=0.000001,lr=args.lr,
                                 betas=(0.9,0.999),weight_decay=0.0001)
    #optimizer = torch.optim.SGD(net.parameters(),lr=args.lr)


    # training loop
    for epoch in range(1, args.epochs):
        accuracy = train(net, train_loader, optimizer)
        if epoch % 100 == 0 and accuracy == 100:
            break

# graph hidden units
#for layer in [1,2]:
#    if layer == 1 or args.net != 'polar':
#        for node in range(args.hid):
#            graph_hidden(net, layer, node)
#            plt.scatter(full_input[:,0],full_input[:,1],
#                        c=1-full_target[:,0],cmap='RdYlBu')
#            plt.savefig('%s%d_%d.png' % (args.net, layer, node))

# graph output unit
graph_output(net)
plt.scatter(full_input[:,0],full_input[:,1],
            c=1-full_target[:,0],cmap='RdYlBu')
plt.savefig('%s_out.png' % args.net)
