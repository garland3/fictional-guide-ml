#!/usr/bin/env python
# coding: utf-8

# # Goal
# * Generate many functions. 
#     * Model the function with an MLP and fit with n epochs
#     * Use the MLP to find $dydx$ and $d2y/d2x$ using `torch.autograd.grad`.
#     * Use the  range of -1 to 1
#     * Calculate the expected $dydx$ and $d2y/d2x$ for each function using sympy
#     * Compare the L2 error between the analytical values and the MLP evaluted values for $dydx$ and $d2y/d2x$ 
#     * Record the error values 
#     * Plot the performance of the MLP to model the function and its deriviates
# 


import torch
import torch.nn as nn
import torch.nn.functional as F
# import matplotlib.pyplot as plt
from  sympy import *
import random
import numpy as np
import warnings
from base import *
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from pathlib import Path

do_step = Stepper()

x = symbols("x")

def add(x1,x2): return x1-x2
def minus(x1, x2): return x1-x2
def power(x1, x2): return x1**x2
def identity(x1): return x1


fns = [cos, sin, identity]
fns2 = [add, minus, power]




def make_equation(number_of_num_fncs):
    y  = 1
    for i in range(number_of_num_fncs):
        if random.randint(0,1)==0:
            # only requires 1 input. 
            fn = random.choice(fns)
            y = y*fn(x)
        else:
            # requires 2 inputs. 
            fn = random.choice(fns2)
            value = random.random()-0.5
            # now choose if x is the input
            if type(y)!=int and  y.has(x): # first, make sure y already has an x in its expression
                if random.randint(0,1)==0:
                    y = y*fn(x,value) # just use x as the input
                else:
                    print(f"making composite, fn is {str(fn)} and  y = {y}")
                    y = fn(y,value) # do a composite function
                    
            else:
                # if missing x in the express, then we need to use X for sure as the 1st input. 
                y = y*fn(x,value)
#         print(i, y)

    # catch the special case where everything cancels out. Just make the identity function
    if type(y)==int or  y.has(x)== False:
        print(f"catching function with nothing {y}")
        y = x
    return y
    



def make_whole_equation(possible_operations = 3, min_operations = 1):
    number_of_num_fncs_numerator = random.randint(min_operations,possible_operations)
    number_of_num_fncs_denominator = random.randint(min_operations,possible_operations)
#     print(number_of_num_fncs_numerator,number_of_num_fncs_denominator )
    numerator = make_equation(number_of_num_fncs_numerator)
    denominator = make_equation(number_of_num_fncs_numerator)
    equation = numerator/denominator
    if type(equation)==int or  equation.has(x)== False:
        print(f"catching function with nothing {equation} within make_whole_equation")
        equation = x
    return equation

def get_y_values(equation, x_numeric):
    """get the y values evalued at x_numeric"""
    f = lambdify(x, equation, "numpy")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = f(x_numeric)
        y = np.nan_to_num(y)

    return y

def get_dydx_values(equation, x_numeric):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    #     dydx = simplify( diff(equation, x))
        dydx =  diff(equation, x)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f_dydx = lambdify(x, dydx, "numpy")
        dydx_values = f_dydx(x_numeric)
        dydx_values = np.nan_to_num(dydx_values)
    return dydx_values

def get_d2yd2x_values(equation, x_numeric):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dydx =  diff(equation, x)
        d2yd2x =  diff(dydx, x)
        f_d2yd2x = lambdify(x, d2yd2x, "numpy")
        d2yd2x_values = f_d2yd2x(x_numeric)
        d2yd2x_values = np.nan_to_num(d2yd2x_values)
    return d2yd2x_values 

@dataclass_json
@dataclass
class Results:
    fn_loss:float
    dydx_loss:float
    d2yd2x_loss:float
    fn:str

def make_file_name(idx, p):
    return p / f"result_{idx}.json"

def parse_file_name(fname):
    i = fname.stem.find("_")
    return int(fname.stem[i+1:])

def get_next_idx(folder):
    files = list(folder.rglob("*.json"))
    idxs = [parse_file_name(f) for f in files]
    if len(idxs)==0: 
        print("no saved results files found. returning 1")
        return 1
    new_idx = max(idxs)+1
    return new_idx


# clip eerything to be in the range of -nn to nn
nn = 30
num_points_to_model = 50
num_equations = 5
folder = 'results'


folder = Path(folder)
folder.mkdir(exist_ok=True)
x_numeric = np.linspace(-1,1, num_points_to_model)
for i in range(num_equations):
    equation = make_whole_equation()
    y = get_y_values(equation, x_numeric)
    dydx_values = get_dydx_values(equation, x_numeric)
    d2yd2x_values = get_d2yd2x_values(equation, x_numeric)


    # make a torch version of the symbolically found dydx  
    my_tensors  = [ dydx_values,d2yd2x_values, y]
    dydx_values_true_t,d2yd2x_values_true_t, y = \
         [torch.clamp(torch.Tensor(v),-nn,nn) for v in my_tensors]


    xb, yb = [make_batch(torch.tensor(np.nan_to_num(z))).to(torch.float32) for z in [x_numeric,y]]
    yb_normalizer = Normalizer(yb)
    yb_norm = yb_normalizer.norm(yb)


    # ## Train the MLP
    # * a learning rate higher than 1e-2 is generally unstable. 
    mlp = make_mlp(n = 100, layers_count=3, act = Mish)
    do_step2 = Stepper_v2(mlp, xb, yb_norm)
    do_step2.do_epochs(int(1e4), lr = 1e-4)
    yprime = mlp(xb)

    yprime_debatch = yb_normalizer.denorm(   debatch(yprime))
    # x_debatch = debatch(x)

    xb.requires_grad = True
    yprime_pre = mlp(xb)
    yprime_pre.retain_grad()
    # yprime_pre.requires_grad = True
    yprime = yb_normalizer.denorm(   yprime_pre)

    dydx = torch.autograd.grad(yprime.sum(), xb, create_graph=True)[0]
    d2yd2x = torch.autograd.grad(dydx.sum(), xb, create_graph=True)[0]

    dydx, d2yd2x = [debatch(v) for v in [dydx, d2yd2x ]]
    my_tensors  = [dydx, d2yd2x]
    dydx, d2yd2x= [torch.clamp(v,-nn,nn) for v in my_tensors]

    dydx_error = F.mse_loss(dydx_values_true_t,debatch(dydx))
    d2yd2x_error = F.mse_loss(d2yd2x_values_true_t, d2yd2x)

    r = Results(do_step2.loss_list[-1].item(), dydx_error.item(), d2yd2x_error.item(), str(equation))
    j = r.to_json()
    new_idx = get_next_idx(folder)
    fname = make_file_name(new_idx,folder)
    with open(fname, 'w') as f:
        f.write(j)
    print(f"Finished {i} with equation {equation}")
    





