#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
(C) 2024 Airbus copyright all rights reserved
Created on Wed Sep 18 17:49:07 2024

@author: roland
"""

import torch
from torch.nn import ReLU, ReLU6, LeakyReLU, Conv2d, BatchNorm2d, Sequential

def replace_LeakyReLU_with_ReLU(model):
    for n, c in model.named_children():
        if isinstance(c, LeakyReLU):
            setattr(model, n, ReLU())
        else:
            replace_LeakyReLU_with_ReLU(c)

def replace_LeakyReLU_with_ReLU6(model):
    for n, c in model.named_children():
        if isinstance(c, LeakyReLU):
            setattr(model, n, ReLU6())
        else:
            replace_LeakyReLU_with_ReLU6(c)
            
def fix_padding(model):
    for n, c in model.named_children():
        if isinstance(c, Conv2d):
            if c.padding == 'same': # Assuming 3x3 kernel
                c.padding = (1, 1)
        else:
            fix_padding(c)

def fuse_bn_sequential(block):
    """
        y = gamma*(conv(x,w)-mu)/sqrt(var+epsilon)+beta
    """
    stack = []
    flag = False
    temp_conv = None
    for m in block.children():
        if isinstance(m, Conv2d):
            temp_conv = m
        elif isinstance(m, BatchNorm2d) and isinstance(temp_conv, Conv2d):
            flag = True
            bn_st_dict = m.state_dict()
            conv_st_dict = temp_conv.state_dict()
            # BatchNorm params
            eps   = m.eps
            mu    = bn_st_dict['running_mean']
            var   = bn_st_dict['running_var']
            if 'weight' in bn_st_dict:
                gamma = bn_st_dict['weight']
            else:
                gamma = torch.tensor(1.0, dtype=torch.float32)
            if 'bias' in bn_st_dict:
                beta  = bn_st_dict['bias']
            else:
                beta = torch.tensor(0.0, dtype=torch.float32)
            # Conv params
            weight = conv_st_dict['weight']
            conv_bias = conv_st_dict.get('bias')
            # fusion the params
            denom = torch.sqrt(var + eps)
            A = gamma.div(denom)
            bias = beta - A.mul(mu)
            if conv_bias is not None:
                bias = bias + A.mul(conv_bias)
            weight = (weight.transpose(0, -1).mul_(A)).transpose(0, -1)
            # assign to the new conv
            temp_conv.weight.data.copy_(weight)
            temp_conv.bias = torch.nn.Parameter(bias)
            stack.append(temp_conv)
            temp_conv = None
        else:
            temp_conv = None
            stack.append(m)
    if temp_conv is not None:
        stack.append(temp_conv)
    if flag:
        return Sequential(*stack)
    else:
        return block

def fuse_bn_recursively(model):
    model = fuse_bn_sequential(model)
    for module_name in model._modules:
        model._modules[module_name] = fuse_bn_recursively(model._modules[module_name])
    return model
