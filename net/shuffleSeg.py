import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import slim
class BasicUnitA(nn.Module):
    def __init__(self, in_c, out_c, group, stride=2, dilation=1):
        super(BasicUnitA, self).__init__()
        self.stride = stride
        out_c -= in_c
        bottleneck = out_c // 4
        assert bottleneck % group == 0
        assert out_c % group == 0
        assert stride == 2

        self.g_conv1 = slim.conv_bn_relu(in_c, bottleneck, 1, stride=1, pad=0, group=group)
        self.shuffle2 = slim.get_shuffle(group)
        self.dw_conv3 = slim.conv_bn(bottleneck, bottleneck, 3, stride=2, pad=1, group=bottleneck)
        self.g_conv4 = slim.conv_bn(bottleneck, out_c, 1, stride=1, pad=0, group=group)

        self.avg_pool1 = nn.AvgPool2d(3, 2, 1)
        print bottleneck

    def forward(self, x):
        print x.size()
        out = self.g_conv1(x)
        out = self.shuffle2(out)
        out = self.dw_conv3(out)
        out1 = self.g_conv4(out)

        out2 = self.avg_pool1(x)
        return F.relu(torch.cat([out1, out2], dim=1))

class BasicUnitB(nn.Module):
    def __init__(self, in_c, out_c, group, stride=1, dilation=1):
        super(BasicUnitB, self).__init__()
        bottleneck = out_c // 4
        assert stride == 1
        #assert in_c == out_c
        assert bottleneck % group == 0
        self.g_conv1 = slim.conv_bn_relu(in_c, bottleneck, 1, stride=stride, pad=0, group=group)
        self.shuffle2 = slim.get_shuffle(group)
        self.dw_conv3 = slim.conv_bn(bottleneck, bottleneck, 3, stride=1, pad=1, group=bottleneck)
        self.g_conv4 = slim.conv_bn(bottleneck, out_c, 1, stride=1, pad=0, group=group)

    def forward(self, x):
        out = self.g_conv1(x)
        out = self.shuffle2(out)
        out = self.dw_conv3(out)
        out = self.g_conv4(out)
        return F.relu(out + x)

if __name__ == '__main__':
    from torch.autograd import Variable
    stage1 = nn.Sequential(BasicUnitA(24, 200, 2),
            BasicUnitB(200, 200, 2),
            BasicUnitB(200, 200, 2),
            BasicUnitB(200, 200, 2),
            )
    x = Variable(torch.rand(2, 24, 64, 64), volatile=False)
    out = stage1(x)
    print out.size()

