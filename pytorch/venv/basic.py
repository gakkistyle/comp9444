import torch

import numpy as np

# we create tensors in a similar way to numpy and arrays
x_numpy = np.array([0.1,0.2,0.3])
x_torch = torch.tensor([0.1,0.2,0.3])
print('x_numpy, x_torch')
print(x_numpy,x_torch)
print()

# to and from numpy, pytorch
print('to and from numpy and pytorch')
print(torch.from_numpy(x_numpy),x_torch.numpy())
print()

# we can do basic operations like +-*/
y_numpy = np.array([3,4,5.])
y_torch = torch.tensor([3,4,5.])
print("x+y")
print(x_numpy + y_numpy,x_torch+y_torch)
print()

# many functions that are in numpy are also in pytorch
print("norm")
print(np.linalg.norm(x_numpy),torch.norm(x_torch))
print()

# to apply an operation along a dimension,
# we use the dim keyword argument instead of axis
print("mean along the 0th dimension")
x_numpy = np.array([[1,2],[3,4.]])
x_torch = torch.tensor([[1,2],[3,4.]])
print(np.mean(x_numpy,axis=0),torch.mean(x_torch,dim=0))
