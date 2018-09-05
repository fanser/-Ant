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

    def forward(self, x):
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

class LayoutNet(nn.Module):
    def __init__(self, num_classes):
        super(LayoutNet, self).__init__()
        group = 2
        self.stage1 = nn.Sequential(
            slim.conv_bn_relu(c_in=3, c_out=24, k_size=3, stride=2, pad=0),
            nn.MaxPool2d(2, 2) 
            )
        self.stage2 = nn.Sequential(BasicUnitA(24, 240, group),
            BasicUnitB(240, 240, group),
            BasicUnitB(240, 240, group),
            )
        self.stage3 = nn.Sequential(BasicUnitA(240, 480, group),
            BasicUnitB(480, 480, group),
            BasicUnitB(480, 480, group),
            BasicUnitB(480, 480, group),
            BasicUnitB(480, 480, group),
            BasicUnitB(480, 480, group),
            BasicUnitB(480, 480, group),
            )
        self.stage4 = nn.Sequential(BasicUnitA(480, 960, group),
            BasicUnitB(960, 960, group),
            BasicUnitB(960, 960, group),
            )

        self.classifier1 = nn.Conv2d(240, num_classes, 1, 1)
        self.classifier2 = nn.Conv2d(480, num_classes, 1, 1)
        self.classifier3 = nn.Conv2d(960, num_classes, 1, 1)
        self.num_classes = num_classes

    def forward(self, imgs, labels):
        n, h, w = labels.shape

        out = self.stage1(imgs)
        out = self.stage2(out)
        preds1 = self.classifier1(out)
        preds1 = F.upsample_bilinear(preds1, size=(h, w))

        out = self.stage3(out)
        preds2 = self.classifier2(out)
        preds2 = F.upsample_bilinear(preds2, size=(h, w))

        out = self.stage4(out)
        preds3 = self.classifier3(out)
        preds3 = F.upsample_bilinear(preds3, size=(h, w))

        #print preds.shape
        preds = preds1 + preds2 + preds3
        loss = F.nll_loss(F.log_softmax(preds), labels)
        return preds, loss, torch.stack([preds1, preds2, preds3])


if __name__ == '__main__':
    from torch.autograd import Variable
    stage1 = nn.Sequential(BasicUnitA(24, 240, 4),
            BasicUnitB(240, 240, 4),
            BasicUnitB(240, 240, 4),
            )
    x = Variable(torch.rand(2, 24, 64, 64), volatile=False)
    out = stage1(x)

