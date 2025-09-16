import torch
from torch import nn
import torch.nn.functional as F
import math


class CDConv2D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                 dilation=1,groups=1,bias=True, theta=0.7):
        super().__init__()
        self.conv = nn.Conv2d(in_ch,out_ch,kernel_size,stride,padding,dilation=dilation,groups=groups,bias=bias)
        self.theta = theta

    def forward(self,x):
        out = self.conv(x)

        if abs(self.theta) < 1e-8:
            return out

        weight = self.conv.weight
        kernel_diff = weight.sum(dim=2,keepdim=True).sum(dim=3,keepdim=True)

        out_diff = F.conv2d(x,kernel_diff,stride=self.conv.stride,padding=0,
                       dilation=self.conv.dilation,groups=self.conv.groups)

        if out_diff.shape[-1]!=out.shape[-1] or out_diff.shape[-2]!=out.shape[-2]:
            out_diff = F.interpolate(out_diff,size=out.shape[-2:],mode="nearest")

        return (1.0 - self.theta)*out + self.theta*(out-out_diff)

        

class CDCppBlock(nn.Module):
    def __init__(self,in_ch,out_ch,stride=1,theta=0.7):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.backbone = nn.Sequential(
            CDConv2D(in_ch,out_ch,kernel_size=3,stride=stride,padding=1,bias=True,theta=theta),
            nn.BatchNorm2d(out_ch),
            self.relu,
            CDConv2D(out_ch,out_ch,kernel_size=3,padding=1,stride=1,bias=True,theta=theta),
            nn.BatchNorm2d(out_ch)
        )
        self.downsample = None
        if stride!=1 or in_ch!=out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch,out_ch,kernel_size=1,stride=stride,bias=True),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self,x):
        identity = x
        out = self.backbone(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out

class CDCN(nn.Module):
    def __init__(self,in_ch=3,stem_out=32,c1=64,c2=128,theta=0.7,use_maxpool=True):
        super().__init__()
        self.stem = nn.Sequential(
            CDConv2D(in_ch,stem_out,kernel_size=7,stride=2,padding=3,bias=True, theta=theta),
            nn.BatchNorm2d(stem_out),
            nn.ReLU(inplace=True)
        )

        self.use_maxpool = use_maxpool
        if self.use_maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.backbone = nn.Sequential(
            CDCppBlock(stem_out,c1,stride=1,theta=theta),
            CDCppBlock(c1,c2,stride=2,theta=theta)
        )

    def forward(self,x):
        B, T, C, H, W = x.shape
        x = x.view(B*T,C,H,W)

        x = self.stem(x)

        if self.use_maxpool:
            x = self.maxpool(x)

        x = self.backbone(x)

        x = x.view(B,T,x.shape[1],x.shape[2],x.shape[3])

        return x

