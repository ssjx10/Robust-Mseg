import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.autograd import Function

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, norm=True, bias=False, drop_rate=0.0):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        
#         self.relu = nn.ReLU(inplace=True) if relu else None
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.drop = nn.Dropout3d(p=drop_rate) if drop_rate != 0 else None
        self.norm = nn.InstanceNorm3d(out_planes) if norm else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        
        x = self.conv(x)
        
        if self.drop is not None:
            x = self.drop(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.relu is not None:
            x = self.relu(x)
        
        return x
    
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class Adaptive_resblock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Adaptive_resblock, self).__init__()
        self.out_channels = out_planes
        
        self.conv1 = BasicConv(in_planes, out_planes, 3, stride=1, padding=(3-1) // 2, relu=False, norm=False)
        self.i_norm1 = Adaptive_instance_norm()
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = BasicConv(in_planes, out_planes, 3, stride=1, padding=(3-1) // 2, relu=False, norm=False)
        self.i_norm2 = Adaptive_instance_norm()

    def forward(self, x_init, mu, sigma):
        
        x = self.conv1(x_init)
        x = self.i_norm1(x, sigma, mu)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.i_norm2(x, sigma, mu)
        
        return x + x_init

class Adaptive_instance_norm(nn.Module):
    def forward(self, content, gamma, beta, epsilon=1e-5):
        c_mean = torch.mean(content, [2,3,4], keepdim=True)
        c_std = torch.std(content, [2,3,4], keepdim=True,)
        
        return gamma * ((content - c_mean) / c_std) + beta


class ZeroLayerF(Function): # instance missing

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        new_x = x.clone()
        new_x[alpha] = 0
        
        return new_x

    @staticmethod
    def backward(ctx, grad_output):
        zero_grad = grad_output.clone() # need a clone!!
        zero_grad[ctx.alpha] = 0 # drop
        
        return zero_grad, None