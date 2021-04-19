import torch
from torchvision import models
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np
from network.ResNet import resnet101,resnet50

# COEFF = 12.0

class Normalize():
    def __init__(self, mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)):

        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img


def _SplitChannels(channels, num_groups):
    split_channels = [channels//num_groups for _ in range(num_groups)]
    split_channels[0] += channels - sum(split_channels)
    return split_channels

class MDConv(nn.Module):
    def __init__(self, channels, kernel_size, stride):
        super(MDConv, self).__init__()

        self.num_groups = len(kernel_size)
        self.split_channels = _SplitChannels(channels, self.num_groups)

        self.mixed_depthwise_conv = nn.ModuleList()
        for i in range(self.num_groups):
            self.mixed_depthwise_conv.append(nn.Conv2d(
                self.split_channels[i],
                self.split_channels[i],
                kernel_size[i],
                stride=stride,
                padding=kernel_size[i]//2,
                groups=self.split_channels[i],
                bias=False
            ))
        for con in self.mixed_depthwise_conv:
            con.weight.data.fill_(0.0)

    def forward(self, x):
        if self.num_groups == 1:
            return self.mixed_depthwise_conv[0](x)

        x_split = torch.split(x, self.split_channels, dim=1)
        x = [conv(t) for conv, t in zip(self.mixed_depthwise_conv, x_split)]
        x = torch.cat(x, dim=1)

        return x




class LIP(nn.Module):
    def __init__(self, channels,COEFF):
        super(LIP, self).__init__()
        self.logit = nn.Sequential(
            OrderedDict((

                ('MDconv', MDConv(channels, [3,5,7],1)),
                ('bn', nn.InstanceNorm2d(channels, affine=True)),

            ))
        )
        self.COEFF = COEFF







    def forward(self, x):

        w = torch.sigmoid(self.logit(x))*self.COEFF
        b, c, _, _ = w.shape
        w = F.softmax(w.view(b,c,-1),dim=2)

        return (x.view(b,c,-1)*w).sum(dim=2).view(b,c,1,1)





class net (nn.Module):
    def __init__(self,COEFF,drop_rate):
        super(net,self).__init__()
        self.backbone = resnet50(pretrained=True)
        self.Lib = LIP(2048,COEFF)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(2048, 20, 1, bias=False)
        self.drop = nn.Dropout(drop_rate)
        # torch.nn.init.xavier_uniform_(self.fc.weight)

        self.normalize = Normalize()




    def forward(self, x,is_eval=False):
        x = self.backbone(x)
        # x = self.Lib(cam)
        feature = x
        x = self.drop(x)
        if(is_eval):
            cam = self.fc(feature)
            return cam,cam
        # x = self.avgpool(x)
        x = self.Lib(x)
        # x = self.avgpool(x)

        x = self.fc(x)
        cam = self.fc(feature)
        return cam,x





if __name__ == '__main__':
    net = net(17.5)
    print(net)
    # print(net.Lib.logit[0].DW_Conv[0].weight)

