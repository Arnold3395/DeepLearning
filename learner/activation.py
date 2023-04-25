import torch

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def tanh(x):
    return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

def hard_tanh(x):
    return torch.max(torch.min(x, torch.tensor(1.0)), torch.tensor(-1.0))

def hard_sigmoid(x):
    return torch.max(torch.min((x + 1) / 2, torch.tensor(1.0)), torch.tensor(0.0))

def relu(x):
    return torch.max(x, torch.tensor(0.0))

def leaky_relu(x):
    return torch.max(x, torch.tensor(0.1) * x)

def prelu(x, alpha=0.1):
    return torch.max(x, alpha * x)

def elu(x, alpha=0.1):
    return torch.max(x, alpha * (torch.exp(x) - 1))

def softplus(x):
    return torch.log(1 + torch.exp(x))

def swish(x):
    return x * sigmoid(x)

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(torch.sqrt(2 / 3.1415926) * (x + 0.044715 * torch.pow(x, 3))))
