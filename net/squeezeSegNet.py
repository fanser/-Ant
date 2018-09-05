import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import slim


__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']


model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}


class Fire1(nn.Module):
    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire1, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

class Fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze_activation = slim.conv_bn_relu(inplanes, squeeze_planes, k_size=1, stride=1, pad=0)
        self.expand1x1_activation = slim.conv_bn_relu(squeeze_planes, expand1x1_planes, k_size=1, stride=1, pad=0)
        self.expand3x3_activation = slim.conv_bn_relu(squeeze_planes, expand3x3_planes, k_size=3, stride=1, pad=1)

    def forward(self, x):
        x = self.squeeze_activation(x)
        return torch.cat([
            self.expand1x1_activation(x),
            self.expand3x3_activation(x)
        ], 1)



class LayoutNet(nn.Module):
    def __init__(self, num_classes = 6, version=1.0):
        super(LayoutNet, self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        self.num_classes = num_classes
        if version == 1.0:
            self.features = nn.Sequential(
                nn.Conv2d(3, 48, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
                Fire(48, 16, 48, 48),
                Fire(96, 32, 48, 48),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
                Fire(96, 32, 64, 64),
                Fire(128, 48, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
                Fire(128, 48, 96, 96),
                Fire(192, 64, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
                Fire(256, 64, 128, 128),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(256, self.num_classes, kernel_size=1)
        self.classifier = final_conv
        self.classifier1 = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            #nn.ReLU(inplace=True),
            #nn.AvgPool2d(13, stride=1)
            #nn.AdaptiveAvgPool2d(1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        #print x.shape
        return x
        #return x.view(x.size(0), self.num_classes)

    def _load_state_dict(self, src_state_dict):
        dst_state_dict = self.state_dict()
        for key, value in dst_state_dict.items():
            if 'classifier' in key:
                continue
            #print "Copying ", key, value.size(), src_state_dict[key].size()
            value[:] = src_state_dict[key]

def seg_forward(model, x, labels):
    preds = model(x)
    #print preds.max()
    n, h, w = labels.shape
    preds = F.upsample_bilinear(preds, size=(h, w))
    loss = F.nll_loss(F.log_softmax(preds), labels)
    return preds, loss


def SegForward(object):
    def __init__(self, num_classes):
        pass

    def __call__(self, x, label):
        preds = self.model(x)
        print preds.shape
        n, h, w = labels.shape
        preds = F.upsample_bilinear(preds, size=(h, w))
        print preds.shape
        loss = F.nll_loss(F.log_softmax(preds), labels)
        return preds, loss

if __name__ == '__main__':
    from torch.autograd import Variable
    model = LayoutNet(6)
    x = Variable(torch.rand(2, 3, 256, 256))
    label = Variable(torch.ones(2, 256, 256).long())
    preds, loss = seg_forward(model, x, label)
    print loss
    loss.backward()
    print loss
