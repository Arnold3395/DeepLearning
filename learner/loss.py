import torch

def cross_entropy(y_pred,y):
    return -torch.sum(y*torch.log(y_pred) + (1-y)*torch.log(1-y_pred))