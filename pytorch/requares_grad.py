import  torch

a = torch.tensor(2.0,requires_grad=True)
b = torch.tensor(1.0,requires_grad=True)
c = a+b
d = b+1
e = c*d
print('c',c)
print('d',d)
print('e',e)
