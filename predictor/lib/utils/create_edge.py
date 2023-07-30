import torch
import numpy as np


def create_1toN_adj(n):
    a = np.zeros((n,n))
    a[0].fill(1)
    a[:,0].fill(1)
    # a[0,0] = 0.0
    return torch.tensor(a, requires_grad=False)

def create_fc_adj(n):
    # return torch.ones(n,n)
    return torch.ones(n,n)-torch.eye(n)