################################################################
# slope.py
# UNSW, COMP9444
# Alan Blair
#
# Pytorch program to perform the simplest possible machine learning task:
#    solve F(x) = A*x such that f(1)=1
#
# (a) Run the code by typing:
#     python3 slope.py --lr 0.1
#     Try different values of learning rate
#    (0.01, 0.1, 0.5, 1.0, 1.5, 1.9, 2.0, 2.1)
#     and explain what happens in each case.
#
# (b) Now add momentum by typing
#     python3 slope.py --mom 0.1
#     Try running with momentum = 0.1, 0.2, ... 1.0, 1.1
#    (keeping the learning rate fixed at 1.9)
#     For which value of momentum does it learn fastest?
#     What happens when the momentum is 1.0? when it is 1.1?

import torch
import torch.utils.data
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1.9, help='learning rate')
parser.add_argument('--mom',type=float, default=0.0, help='momentum')

args = parser.parse_args()

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.A = torch.nn.Parameter(torch.zeros((1), requires_grad=True))
    def forward(self, input):
        output  = self.A * input
        return(output)

device = 'cpu'

input  = torch.Tensor([[1]])
target = torch.Tensor([[1]])

slope_dataset = torch.utils.data.TensorDataset(input,target)
train_loader  = torch.utils.data.DataLoader(slope_dataset,batch_size=1)

# create neural network according to model specification
net = MyModel().to(device) # CPU or GPU

# choose between SGD, Adam or other optimizer
optimizer = torch.optim.SGD(net.parameters(),lr=args.lr,momentum=args.mom)

epochs = 10000

for epoch in range(1, epochs):
    for batch_id, (data,target) in enumerate(train_loader):
        optimizer.zero_grad() # zero the gradients
        output = net(data)    # apply network
        loss = 0.5*torch.mean((output-target)*(output-target))
        if type(net.A.grad) == type(None):
            print('Ep%3d: zero_grad(): A.grad=  None  A.data=%7.4f loss=%7.4f' \
                      % (epoch, net.A.data, loss))
        else:
            print('Ep%3d: zero_grad(): A.grad=%7.4f A.data=%7.4f loss=%7.4f' \
                      % (epoch, net.A.grad, net.A.data, loss))
        loss.backward()       # compute gradients
        optimizer.step()      # update weights
        print('            step(): A.grad=%7.4f A.data=%7.4f' \
                      % (net.A.grad, net.A.data))
        if loss < 0.000000001 or np.isnan(loss.data):
            exit()
