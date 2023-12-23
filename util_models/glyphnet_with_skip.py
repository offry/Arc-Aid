import cv2
import kornia.filters
import torch
import torchvision.transforms.functional

from utils_functions.imports import *

from util_models.unet_model import *
from util_models.unet_parts import *


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.up1 = Up(512, 256 // 1, False)
        self.up2 = Up(256, 128 // 1, False)
        self.up3 = Up(128, 64 // 1, False)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.last_conv = conv3x3(64, 1)

class Glyphnet_with_skip(nn.Module):
    def __init__(self, model):
        super(Glyphnet_with_skip, self).__init__()
        self.model = model
        self.decoder = Decoder()

    def forward_pred(self, image):
        identity = image

        x1 = self.model.first_block(image)
        x2 = torch.nn.Sequential(self.model.inner_blocks[0])(x1)
        x3 = torch.nn.Sequential(self.model.inner_blocks[1])(x2)
        x4 = torch.nn.Sequential(self.model.inner_blocks[2])(x3)
        x5 = torch.nn.Sequential(self.model.inner_blocks[3])(x4)

        x6 = F.relu(self.model.final_block.bn(self.model.final_block.sconv(x5)))
        pred_net = torch.mean(x6, dim=(-1, -2))
        pred_net = self.model.final_block.softmax(self.model.final_block.fully_connected(self.model.final_block.dropout(pred_net)))
        return pred_net, x1, x2, x3, x4, x5, x6

    def forward_decode(self, x1, x2, x3, x4, x5, x6 ):
        reconst0 = self.decoder.up1(x6, x4)
        reconst1 = self.decoder.up2(reconst0, x3)
        reconst2 = self.decoder.up(reconst1)
        reconst3 = self.decoder.up3(reconst2, x1)
        reconst4 = self.decoder.up(reconst3)
        reconst5 = self.decoder.up(reconst4)
        reconst5 = self.decoder.last_conv(reconst5)
        reconst = torchvision.transforms.functional.invert(reconst5)
        return reconst

    def forward(self, image):
        pred, x1, x2, x3, x4, x5, x6 = self.forward_pred(image)
        reconst = self.forward_decode(x1, x2, x3, x4, x5, x6)
        return pred, reconst
