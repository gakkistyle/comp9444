import torch
import numpy as np
def f(x):
    return (x-2)**2

def fp(x):
    return 2*(x-2)

x = torch.tensor([1.0],requires_grad=True)

y = f(x)
y.backward()

print('Analytical f\'(x):',fp(x))
print('PyTorch\'s f\'(x):',x.grad)

def g(w):
    return 2*w[0]*w[1] + w[1]*torch.cos(w[0])

def grad_g(w):
    return torch.tensor([2*w[1] - w[1]*torch.sin(w[0]),2*w[0]+torch.cos(w[0])])

w = torch.tensor([np.pi,1],requires_grad=True)

z = g(w)
z.backward()

print('Analytical grad g(w):',grad_g(w))
print('PyTorch\'s grad g(w):',w.grad)

x = torch.tensor([5.0],requires_grad=True)
step_size = 0.25

print('iter,\tx,\tf(x),\tf\'(x),\tf\'(x) pytorch')
for i in range(15):
    y = f(x)
    y.backward()

    print('{},\t{:.3f},\t{:.3f}\t{:.3f}\t{:.3f}'.format(i,x.item(),f(x).item(),fp(x).item(),x.grad.item()))

    x.data = x.data - step_size*x.grad

    x.grad.detach()
    x.grad.zero_()

