# encoder_model.py
# COMP9444, CSE, UNSW

import torch
import matplotlib.pyplot as plt

class EncModel(torch.nn.Module):
    # fully connected two-layer network
    def __init__(self, num_input, num_hid, num_out):
        super(EncModel, self).__init__()
        self.in_hid  = torch.nn.Linear(num_input, num_hid)
        self.hid_out = torch.nn.Linear(num_hid, num_out)
    def forward(self, input):
        hid_sum = self.in_hid(input)
        hidden  = torch.tanh(hid_sum)
        out_sum = self.hid_out(hidden)
        output  = torch.sigmoid(out_sum)
        return(output)

def plot_hidden(net):
    # plot the hidden unit dynamics of the network
    plt.xlim(-1,1), plt.ylim(-1,1) # limits of x and y axes

    # input to hidden weights and biases
    weight = net.in_hid.weight.data.cpu()
    bias   = net.in_hid.bias.data.cpu()

    num_in  = net.in_hid.weight.data.size()[1]
    num_out = net.hid_out.weight.data.size()[0]

    # draw a dot to show where each input is mapped to in hidden unit space
    P = torch.tanh(weight + bias.unsqueeze(1).repeat(1,num_in))
    plt.plot(P[0,:],P[1,:],'bo')

    # draw a line interval to show the decision boundary of each output
    for i in range(num_out):

        A = net.hid_out.weight.data.cpu()[i,0]
        B = net.hid_out.weight.data.cpu()[i,1]
        C = net.hid_out.bias.data.cpu()[i]

        j = 0;
        if A == 0:
            if B != 0:
                y0 = -C/B
                if -1 < y0 and y0 < 1:
                    j = 2
                    plt.plot([-1,1],[y0,y0])
        elif B == 0:
            if A != 0:
                x0 = -C/A
                if -1 < x0 and x0 < 1:
                    plt.plot([x0,x0],[-1,1])
        else:
            x = torch.zeros(2)
            y = torch.zeros(2)
            y0 = (A-C)/B
            if -1 <= y0 and y0 <= 1:
                x[j] = -1
                y[j] =  y0
                j = j+1
            y0 = (-A-C)/B
            if -1 <= y0 and y0 <= 1:
                x[j] =  1
                y[j] =  y0
                j = j+1
            x0 = (B-C)/A
            if j < 2 and -1 <= x0 and x0 <= 1:
                x[j] =  x0
                y[j] = -1
                j = j+1
            x0 = (-B-C)/A
            if j < 2 and -1 <= x0 and x0 <= 1:
                x[j] =  x0
                y[j] =  1
                j = j+1
            if j > 1:
                plt.plot(x,y)
