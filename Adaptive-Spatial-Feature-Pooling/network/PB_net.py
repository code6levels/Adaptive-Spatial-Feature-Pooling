import torch
import torch.nn as nn
import torchvision
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
from sync_batchnorm import convert_model
from torchvision import models

__all__ = ['ResNet50', 'ResNet101','ResNet152']

COEFF = 12

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

def lip2d(x, logit, kernel=3, stride=2, padding=1):
    weight = logit.exp()
    return F.avg_pool2d(x*weight, kernel, stride, padding)/F.avg_pool2d(weight, kernel, stride, padding)

class SoftGate(nn.Module):
    def __init__(self,COEFF):
        super(SoftGate, self).__init__()
        self.COEFF = COEFF

    def forward(self, x):
        return torch.sigmoid(x).mul(self.COEFF)

class AP(nn.Module):
    def __init__(self, channels,kernel_group,COEFF,kernel_size=3, stride=2, padding=1):
        super(AP, self).__init__()
        self.logit = nn.Sequential(
            OrderedDict((
                ('MDconv', MDConv(channels, kernel_group,1)),
                ('bn', nn.InstanceNorm2d(channels, affine=True)),
                ('gate', SoftGate(COEFF)),
            ))
        )
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        frac = lip2d(x, self.logit(x),self.kernel_size, self.stride, self.padding)
        return frac

class AP_gap(nn.Module):
    def __init__(self, channels,kernel_group,COEFF):
        super(AP_gap, self).__init__()
        self.logit = nn.Sequential(
            OrderedDict((
                ('MDconv', MDConv(channels, kernel_group,1)),
                ('bn', nn.InstanceNorm2d(channels, affine=True)),
            ))
        )
        self.COEFF = COEFF
    def forward(self, x):
        mask = self.logit(x)
        b, c, h, w = mask.shape
        w = torch.sigmoid(mask)*self.COEFF
        w = F.softmax(w.view(b,c,-1),dim=2)
        return (x.view(b,c,-1)*w).sum(dim=2).view(b,c,1,1)

class Bottleneck(nn.Module):
    def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling
        self.conv1 = nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(places)
        self.conv2 = nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 =  nn.BatchNorm2d(places)
        self.conv3 = nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(places*self.expansion)
        self.relu = nn.ReLU(inplace=True)

        if self.downsampling:
            if stride == 1:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(places*self.expansion)
                )
            else:
                self.downsample = nn.Sequential(
                    OrderedDict((
                        ('AP_pool',AP(in_places,[3],COEFF,2,2,0)),
                        ('0', nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False)),
                        ('1',nn.BatchNorm2d(places*self.expansion)),
                    ))
                )


    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self,blocks, expansion = 4,GAP_COEFF=16):
        super(ResNet,self).__init__()
        self.expansion = expansion

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = AP(64,[3],COEFF)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.layer1 = self.make_layer(in_places = 64, places= 64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places = 256,places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512,places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024,places=512, block=blocks[3], stride=2)

        self.AP_gap= AP_gap(2048, [3,5,7],GAP_COEFF)
        self.FC = nn.Conv2d(2048, 20, 1, bias=False)
        self.normalize = Normalize()


    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)


    def forward(self, x,is_eval=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feature = x
        if (is_eval):
            cam = self.FC(feature)
            return cam, cam

        x = self.AP_gap(x)
        x = self.FC(x)
        cam = self.FC(feature)
        return cam, x


def ResNet50(GAP_COEFF=16):
    res50 = models.resnet50(pretrained=True)
    model = ResNet([3, 4, 6, 3],GAP_COEFF=GAP_COEFF)
    model.load_state_dict(res50.state_dict(),strict=False)
    # model = convert_model(model)

    return model



def ResNet101():
    res50 = models.resnet50(pretrained=True)
    model = ResNet([3, 4, 23, 3])
    model.load_state_dict(res50.state_dict(), strict=False)
    model = convert_model(model)

    return model

def ResNet152():
    return ResNet([3, 8, 36, 3])


if __name__=='__main__':

    model_2 = ResNet50()


    input = torch.randn(1, 3, 224, 224)

    out,_ = model_2(input)
    print(out.shape)
