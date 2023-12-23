import cv2
import kornia.filters
import torch
import torchvision.transforms.functional

from utils_functions.imports import *

from util_models.unet_model import *
from util_models.unet_parts import *
import scipy.ndimage


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv0 = conv3x3(384, 272)
        self.up0 = Up(272, 136// 1, False)
        self.conv1 = conv3x3(136, 96)
        self.up1 = Up(96, 48 // 1, False)
        self.up2 = Up(48, 24 // 1, False)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.last_conv = conv3x3(24, 1)

class Efficientnet_with_skip(nn.Module):
    def __init__(self, model):
        super(Efficientnet_with_skip, self).__init__()
        self.model = model
        self.decoder = Decoder()

    def forward_pred(self, image):
        pred_net = self.model(image)
        return pred_net

    def forward_decode(self, image):
        identity = image
        x1 = torch.nn.Sequential(self.model.features[0])(image)
        x2 = torch.nn.Sequential(self.model.features[1])(x1)
        x3 = torch.nn.Sequential(self.model.features[2])(x2)
        x4 = torch.nn.Sequential(self.model.features[3])(x3)
        x5 = torch.nn.Sequential(self.model.features[4])(x4)
        x6 = torch.nn.Sequential(self.model.features[5])(x5)
        x7 = torch.nn.Sequential(self.model.features[6])(x6)
        x8 = torch.nn.Sequential(self.model.features[7])(x7)

        reconst0 = F.leaky_relu(self.decoder.conv0(x8))
        reconst0 = self.decoder.up0(reconst0, x6)
        reconst0 = F.leaky_relu(self.decoder.conv1(reconst0))
        reconst1 = self.decoder.up1(reconst0, x4)
        reconst1 = self.decoder.up(reconst1)
        reconst2 = self.decoder.up2(reconst1, x2)
        reconst2 = self.decoder.up(reconst2)
        reconst = self.decoder.last_conv(reconst2)
        reconst = torchvision.transforms.functional.invert(reconst)
        return reconst

    def forward(self, image):
        reconst = self.forward_decode(image)
        pred = self.forward_pred(image)
        return pred, reconst
