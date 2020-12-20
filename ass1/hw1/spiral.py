# spiral.py
# COMP9444, CSE, UNSW

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        # INSERT CODE HERE
        self.linear1 = nn.Linear(2, num_hid)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(num_hid,1)
        self.sigmoid = nn.Sigmoid()
        self.hid1 = 0

    def forward(self, input):
        #output = 0*input[:,0] # CHANGE CODE HERE
        x = input[:,0]
        y = input[:,1]
        r = torch.sqrt(torch.pow(x,2) + torch.pow(y,2)).view(-1,1)
        
        a = torch.atan2(y,x).view(-1,1)
        #print(a)
        x_1 = torch.cat((r,a),1)
        self.hid1 = self.tanh(self.linear1(x_1))
        output = self.sigmoid(self.linear2(self.hid1))
        return output


class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        # INSERT CODE HERE
        self.linear1 = nn.Linear(2,num_hid)
        self.tanh = nn.Tanh()
        #self.relu = nn.ReLU()
        self.linear2 = nn.Linear(num_hid,num_hid)
        #self.linear3 = nn.Linear(num_hid,num_hid)
        self.output = nn.Linear(num_hid,1)
        self.sigmoid = nn.Sigmoid()
        self.hid1 = 0
        self.hid2 = 0

    def forward(self, input):
        #output = 0*input[:,0] # CHANGE CODE HERE
        x = input[:,0].view(-1,1)
        y = input[:,1].view(-1,1)
        #print(x)
        x_1 = torch.cat((x,y),1)
        self.hid1 = self.tanh(self.linear1(x_1))
        self.hid2 = self.tanh(self.linear2(self.hid1))
        # self.hid1 = self.relu(self.linear1(x_1))
        # self.hid2 = self.relu(self.linear2(self.hid1))
        #self.hid2 = self.tanh(self.linear3(self.tanh(self.linear2(self.hid1))))
        output = self.sigmoid(self.output(self.hid2))
        return output

def graph_hidden(net, layer, node):
    #plt.clf()
    # INSERT CODE HERE
    xrange = torch.arange(start=-7,end=7.1,step=0.01,dtype=torch.float32)
    yrange = torch.arange(start=-6.6,end=6.7,step=0.01,dtype=torch.float32)
    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    grid = torch.cat((xcoord.unsqueeze(1),ycoord.unsqueeze(1)),1)

    with torch.no_grad(): # suppress updating of gradients
        net.eval()        # toggle batch norm, dropout
        net(grid)
        
        if layer == 1:
            pred = (net.hid1[:,node] >= 0.5).float()
        elif layer == 2:
            pred = (net.hid2[:,node] >= 0.5).float()

        #net.train() # toggle batch norm, dropout back again


        # plot function computed by model
        plt.clf()
        plt.pcolormesh(xrange,yrange,pred.cpu().view(yrange.size()[0],xrange.size()[0]), cmap='Wistia')
