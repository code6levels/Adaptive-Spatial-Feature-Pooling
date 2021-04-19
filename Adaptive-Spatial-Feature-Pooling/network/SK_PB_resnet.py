import torch
from torchvision import models
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from sync_batchnorm import convert_model
import numpy as np
# from network.lip_resnet import resnet50

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


class SKConv(nn.Module):
    def __init__(self, features, WH, M, G, r, stride=1, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            k = 3 + i * 2
            self.convs.append(
                nn.Conv2d(features, features, kernel_size=3 + i * 2, stride=stride, padding=1 + i, groups=G,bias=False),
            )
        # self.gap = nn.AvgPool2d(int(WH/stride))
        for con in self.convs:
            con.weight.data.fill_(0.0)
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)
        self.IN = nn.InstanceNorm2d(features,affine=True)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        # fea_s = self.gap(fea_U).squeeze_()
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        fea_v = self.IN(fea_v)
        return fea_v

class LIP(nn.Module):
    def __init__(self, channels,COEFF):
        super(LIP, self).__init__()
        # self.logit = nn.Sequential(
        #     OrderedDict((
        #
        #         ('MDconv', MDConv(channels, [3,5],1)),
        #         ('bn', nn.InstanceNorm2d(channels, affine=True)),
        #
        #     ))
        # )
        self.logit = SKConv(2048,49,2,2048,2)
        self.COEFF = COEFF






    def forward(self, x):



        # weight = F.softmax(mask_merge.view(b, -1), dim=1).view(b, 1, h, w)

        w = torch.sigmoid(self.logit(x))*self.COEFF
        b, c, _, _ = w.shape

        # mask = self.gate(mask_merge)
        # frac = lip2d(x,mask)
        # # mask = self.gate(self.bn(self.con3(x)+self.con5(x)+self.con7(x)))
        # # frac = lip2d(x,mask)
        # w = F.sigmoid(self.logit(x))*self.COEFF
        # w = self.logit(x)
        w = F.softmax(w.view(b,c,-1),dim=2)
        # x = self.drop(x.view(b,c,-1)*w)
        #
        # return x.sum(dim=2).view(b, c, 1, 1)
        return (x.view(b,c,-1)*w).sum(dim=2).view(b,c,1,1)

    def merge(self,x1,x2,x3):
        x1_split = torch.chunk(x1, 2048, 1)
        x2_split = torch.chunk(x2, 2048, 1)
        x3_split = torch.chunk(x3, 2048, 1)
        out = []
        for i in range(2048):
            out.append(torch.stack((x1_split[i].squeeze(1), x2_split[i].squeeze(1), x3_split[i].squeeze(1)), dim=1))
        out = torch.cat(tuple(out), dim=1)
        return out




class net (nn.Module):
    def __init__(self,COEFF):
        super(net,self).__init__()
        res50 = models.resnet50(pretrained=True)
        # res50 = convert_model(res50)
        # res50 = models.resnet50()
        # res50.layer4[0].conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        # res50.layer4[0].downsample[0] = nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer0 = nn.Sequential(OrderedDict((('conv1',res50.conv1),('bn1',res50.bn1),\
                                                 ('relu',res50.relu),('maxpool',res50.maxpool))))
        self.layer1 = res50.layer1
        self.layer2 = res50.layer2
        self.layer3 = res50.layer3
        self.layer4 = res50.layer4





        self.Lib = LIP(2048,COEFF)
        # self.bn = nn.BatchNorm2d(2048)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(2048, 20, 1, bias=False)
        # torch.nn.init.xavier_uniform_(self.fc.weight)

        self.normalize = Normalize()




    def forward(self, x,is_eval=False):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.drop(x)
        x = self.layer4(x)
        # x = F.relu(self.bn(x))
        # x = self.drop(x)
        # x = self.Lib(cam)
        feature = x
        if(is_eval):
            cam = self.fc(feature)
            return cam,cam
        # x = self.avgpool(x)
        x = self.Lib(x)

        x = self.fc(x)
        cam = self.fc(feature)
        return cam,x





if __name__ == '__main__':
    net = net(17.5)
    print(net)
    # print(net.Lib.logit[0].DW_Conv[0].weight)

