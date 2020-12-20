import torch

d = 2
n = 50
X = torch.randn(n,d)
true_W = torch.tensor([[-1.0],[2.0]])
y = X @ true_W + torch.randn(n,1)*0.1
print(X.shape)
print(y.shape)
print(true_W.shape)

def model(X,w):
    return X @ w

def rss(y,y_hat):
    return torch.norm(y-y_hat)**2/n

def grad_rss(X,y,w):
    # .t() 转置
    return -2*X.t() @ (y - X @ w) / n

w = torch.tensor([[-1.],[0]],requires_grad=True)
y_hat = model(X,w)

loss = rss(y,y_hat)
loss.backward()

print('Analytical gradient',grad_rss(X,y,w).detach().view(2))
print('pytorch gradient',w.grad.view(2))

step_size = 0.1

linear_module = torch.nn.Linear(d,1,bias=False)

loss_func = torch.nn.MSELoss()

optim = torch.optim.SGD(linear_module.parameters(),lr = step_size)

print('iter,\tloss,\tw')

for i in range(20):
    y_hat = linear_module(X)
    loss = loss_func(y_hat,y)
    optim.zero_grad()
    loss.backward()
    optim.step()

    print('{},\t{:.2f},\t{}'.format(i,loss.item(),linear_module.weight.view(2).detach().numpy()))

print('\ntrue w\t\t',true_W.view(2).numpy())
print('estimated w\t',linear_module.weight.view(2).detach().numpy())

