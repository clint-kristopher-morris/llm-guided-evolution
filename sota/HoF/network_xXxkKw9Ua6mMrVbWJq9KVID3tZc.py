
import collections
import argparse

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import *
from torch.optim import SGD

# https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
#from ranger import RangerQH  
#from ranger import RangerVA  
#from ranger import Ranger
#from ranger21 import Ranger21

# --OPTION--
import math
import torch
import torch.optim.lr_scheduler as lr_scheduler

class ConstantMomentumScheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_max, final_momentum):
        self.T_max = T_max
        self.final_momentum = final_momentum
        super().__init__(optimizer)

    def get_momentum(self):
        if self.last_epoch < self.T_max:
            return self.base_momentum - ((self.base_momentum - self.final_momentum) / self.T_max) * self.last_epoch
        else:
            return self.final_momentum

def get_optimizer(model, lr, weight_decay=0, nesterov=True, T_max=None, final_momentum=0.9):
    if weight_decay != 0:
        g0, g1, g2 = [], [], []  # optimizer parameter groups
        for v in model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, torch.nn.Parameter):  # bias
                g2.append(v.bias)
            if isinstance(v, (torch.nn.BatchNorm2d, torch.nn.LayerNorm)):  # weight (no decay)
                g0.append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, torch.nn.Parameter):  # weight (with decay)
                g1.append(v.weight)
        
        opt = torch.optim.SGD(g0, lr, momentum=final_momentum, nesterov=nesterov)
        opt.add_param_group({'params': g1, 'weight_decay': weight_decay})  # add g1 with weight_decay
        opt.add_param_group({'params': g2})  # add g2 (biases)

        if T_max is not None:
            scheduler = ConstantMomentumScheduler(opt, T_max, final_momentum)
            return opt, scheduler
        else:
            return opt
    else:
        opt = torch.optim.SGD(model.parameters(), lr, momentum=final_momentum, nesterov=nesterov)
        if T_max is not None:
            scheduler = ConstantMomentumScheduler(opt, T_max, final_momentum)
            return opt, scheduler
        else:
            return opt
# --OPTION--
import torch.nn as nn

class SE_LN(nn.Module):
    def __init__(self, cin, dropout_p=0.1):
        super().__init__()
        self.gavg = nn.AdaptiveAvgPool2d((1,1))
        self.ln = nn.LayerNorm(cin)
        self.act = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        y = x
        x = self.gavg(x)
        x = x.view(-1, x.size(1))
        x = self.ln(x)
        x = self.act(x)
        x = self.dropout(x)
        x = x.view(-1, x.size(1), 1, 1)
        return x * y
# --OPTION--
import torch.nn as nn

class SE_LN(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.gavg = nn.AdaptiveAvgPool2d((1,1))  
        self.ln = nn.LayerNorm(cin)
        self.act = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)  # Added dropout layer with a drop probability of 0.1

    def forward(self, x):
        y = x
        x = self.gavg(x)
        x = x.view(-1, x.size(1))
        x = self.ln(x)
        x = self.act(x)
        x = self.dropout(x)  # Apply dropout to the output of the activation function
        x = x.view(-1, x.size(1), 1, 1)
        return x*y
# --OPTION--

# -- NOTE --
# Note: The classes SE_LN and SE used in this architecture are pre-existing and fully implemented elsewhere. 
# It is not necessary to create new implementations or modify these classes for this architecture. They should be used as-is. 
# -- NOTE --

def pad_num_x(k_s):
    pad_per_side = int((k_s-1)*0.5)
    return pad_per_side
    
class DFSEBV2(nn.Module):
    def __init__(self, cin, dw_s, is_LN):
        super().__init__()
        self.pw1 = nn.Conv2d(cin, cin, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.act1 = nn.SiLU()
        self.dw1 = nn.Conv2d(cin, cin, dw_s, 1, pad_num_x(dw_s), groups=cin)
        if is_LN:
            self.seln = SE_LN(cin)
        else:
            self.seln = SE(cin,3)
            
        self.pw2 = nn.Conv2d(cin, cin, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cin)
        self.act2 = nn.Hardswish()
        self.dw2 = nn.Conv2d(cin, cin, dw_s, 1, pad_num_x(dw_s), groups=cin)

    def forward(self, x):
        y = x
        x = self.pw1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dw1(x)
        x = self.seln(x)
        x += y
        
        x = self.pw2(x)       
        x = self.bn2(x)
        x = self.act2(x)
        x = self.dw2(x)
        x += y
        return x
        
# --OPTION--
import torch.nn.functional as F

class MinPool2d_y(nn.Module):
    def __init__(self, ks, ceil_mode):
        super().__init__()
        self.ks = ks
        self.ceil_mode = ceil_mode

    def forward(self, x):
        return -F.max_pool2d(-x, self.ks, ceil_mode=self.ceil_mode)

class FCT(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 4, 2, 1, groups=cin, bias=False)
        self.maxpool = nn.MaxPool2d(2, ceil_mode=True)
        self.minpool = MinPool2d_y(2, ceil_mode=True)
        self.pw = nn.Conv2d(3*cin, cout, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.swish = nn.SiLU() # Add swish activation function

    def forward(self, x):
        x = torch.cat((
            self.maxpool(x),
            self.minpool(x),
            self.swish(self.dw(x)), # Add swish activation function
        ), 1)
        x = self.pw(x)
        x = self.swish(self.bn(x)) # Add swish activation function
        return x
# --OPTION--
import torch.nn as nn
import torch.nn.functional as F

class MinPool2d_x(nn.Module):
    def __init__(self, ks, ceil_mode):
        super().__init__()
        self.ks = ks
        self.ceil_mode = ceil_mode

    def forward(self, x):
        return -F.max_pool2d(-x, self.ks, ceil_mode=self.ceil_mode)

class EVE(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2, ceil_mode=True)
        self.minpool = MinPool2d_x(2, ceil_mode=True)
        self.pw = nn.Conv2d(cin*2, cout, 1, 1, bias=False) # Change 1
        self.activation = nn.ReLU() # Add 2

    def forward(self, x):
        x = torch.cat((
            self.maxpool(x),
            self.minpool(x)
        ), 1)
        x = self.pw(x)
        x = self.activation(x) # Add 2
        return x
# --OPTION--
import torch.nn as nn

class ME(nn.Module):
    def __init__(self, cin, cout, dropout_rate=0.25):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2, ceil_mode=True)
        self.pw = nn.Conv2d(cin, cout, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout, affine=True)  # Adding affine to enable bias
        self.act = nn.ReLU()  # Adding ReLU activation
        self.dropout = nn.Dropout2d(dropout_rate)  # Adding dropout layer

    def forward(self, x):
        x = self.maxpool(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)  # Adding activation
        x = self.dropout(x)  # Adding dropout
        return x
# --OPTION--
def pad_num_y(k_s):
    pad_per_side = int((k_s-1)*0.5)
    return pad_per_side
    
class DW(nn.Module):
    def __init__(self, cin, dw_s):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, dw_s, 1, pad_num_y(dw_s), groups=cin)
        self.act = nn.Hardswish()

    def forward(self, x):
        x = self.dw(x)
        x = self.act(x)
        return x

# --OPTION--
# -- NOTE --
# Note: The classes FCT, EVE, ME, and DFSEBV2 used in this architecture are pre-existing and fully implemented elsewhere. 
# It is not necessary to create new implementations or modify these classes for this architecture. They should be used as-is.# -- NOTE --
class ExquisiteNetV2(nn.Module):
    def __init__(self, class_num, img_channels):
        super().__init__()
        self.FCT = FCT(img_channels, 24)
        self.DFSEB1 = DFSEBV2(24, 3, True)
        self.EVE = EVE(24, 72)
        self.DFSEB2 = DFSEBV2(72, 3, True)
        self.ME3 = ME(72, 144)
        self.DFSEB3 = DFSEBV2(144, 3, True)
        self.ME4 = ME(144, 288)
        self.DFSEB4 = DFSEBV2(288, 3, True)
        self.DW = DW(288, 3)
        self.gavg = nn.AdaptiveAvgPool2d((1,1))
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(288, class_num)

    def forward(self, x):
        x = self.FCT(x)
        x = self.DFSEB1(x)
        x = self.EVE(x)
        x = self.DFSEB2(x)
        x = self.ME3(x)
        x = self.DFSEB3(x)
        x = self.ME4(x)
        x = self.DFSEB4(x)
        x = self.DW(x)
        x = self.gavg(x)
        x = self.drop(x)
        x = x.view(-1, x.size(1))
        x = self.fc(x)
        return x
