import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from fastcore.basics import store_attr
from torch.optim import *


# from  sympy import *
# import random
# import numpy as np
# import warnings

def mish(input):
    return input * torch.tanh(F.softplus(input))

class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, _input):
        return mish(_input)
    
class Sine(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, _input):
        return torch.sin(_input)
    
def make_batch(x): return x[:,None]

def debatch(x, detach = True):
    x =  x.squeeze()
    if detach: x = x.detach()
    return x

def make_mlp(n = 20,in_features = 1, out_features = 1, layers_count = 3, act = Mish):
    layers = []
    layers.append(nn.Linear(in_features,n))
    layers.append(act())
    for i in range(layers_count -2):
        layers.append(nn.Linear(n,n))
        layers.append(act())
    layers.append(nn.Linear(n,out_features))
    mlp = nn.Sequential(*layers)
    return mlp

class Stepper:
    clear_grad = True
    def __call__(self,my_mlp,xb, yb, lr = 1e-1):
        yprime = my_mlp(xb)
        self.loss = F.mse_loss(yb,yprime )
        self.loss.backward()
        for name, param in my_mlp.named_parameters():
            param.data = param.data - param.grad*lr
        if self.clear_grad: my_mlp.zero_grad()
        self.yprime = yprime
        return yprime.squeeze().detach()
    
class Stepper_v2:
    """Uses Adam optimizer rather than just SGD. Adam is much better. """
    def __init__(self, my_mlp, xb, yb, lr = 0.001):
        store_attr("my_mlp, xb, yb, lr")
        
    def do_epochs(self, epochs, lr = None):
        lr = self.lr if lr is None else lr
        self.optimizer = Adam(self.my_mlp.parameters(), lr = lr)
        self.loss_list = []
        for i in range(epochs):
            self.optimizer.zero_grad()
            self.yprime = self.my_mlp(self.xb)
            self.loss = F.mse_loss(self.yb,self.yprime )
            self.loss.backward()
            self.optimizer.step()
            self.loss_list.append(self.loss.detach())
            
   
    
class Normalizer:
    def __init__(self, values):
        self.mean = values.mean()
        self.std = values.std()
    
    def norm(self, values):
        return  (values - self.mean)/self.std
    
    def denorm(self, values):
        return values*self.std + self.mean