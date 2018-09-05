from __future__ import division


import numpy as np
import torch
import torch.nn as nn


class Shuffle(nn.Module):
    def __init__(self, groups):
        super(Shuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        n, c, h, w = x.size()
        x = x.view(n, self.groups, c // self.groups, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(n, c, h, w)
        return x

def get_shuffle(groups):
    return Shuffle(groups)


def conv_bn(c_in, c_out, k_size, stride, pad, dilation=1, group=1):
    return nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=k_size, stride=stride, padding=pad, dilation=dilation, groups=group, bias=False),
            nn.BatchNorm2d(c_out)
            )

def conv_bn_relu(c_in, c_out, k_size, stride, pad, dilation=1, group=1):
    return nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=k_size, stride=stride, padding=pad, dilation=dilation, groups=group, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True)
            )

def conv(c_in, c_out, k_size, stride, pad, dilation=1, group=1):
    return nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=k_size, stride=stride, padding=pad, dilation=dilation, groups=group, bias=True)



if __name__ == '__main__':
    net = Shuffle(2)
    x = torch.rand(1, 4, 4, 4)
    y = net(x)
    conv = conv_bn(4, 5, 3, 1, 0)
    z = conv(y)
    
