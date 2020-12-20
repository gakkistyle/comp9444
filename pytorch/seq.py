import torch
from torch import nn
d_in = 3
d_hid = 4
d_out = 1

model = torch.nn.Sequential(
    nn.Linear(d_in,d_hid),
    nn.Tanh(),
    nn.Linear(d_hid,d_out),
    nn.Sigmoid()
)

example = torch.tensor([[1.,2,3],[4,5,6],[4,5,6]])
trans = model(example)
print('transformed',trans)