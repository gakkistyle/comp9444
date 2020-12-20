import torch

# MNIST
N,C,W,H = 10000,3,28,28
X = torch.randn((N,C,W,H))

print(X.shape)
print(X.view(N,C,784).shape)
print(X.view(-1,C,784).shape)

Y = torch.randn((3,4,5,6))
print(Y)

x = torch.empty(5,1,4,1)
y = torch.empty(  3,1,1)
print((x+y).size())