import cv2
import kornia.filters
import torch
import torchvision.transforms.functional

from utils_functions.imports import *

from util_models.unet_model import *
from util_models.unet_parts import *
import scipy.ndimage

class GaussianLayer(nn.Module):
    def __init__(self):
        super(GaussianLayer, self).__init__()
        self.seq = nn.Sequential(
            # nn.ReflectionPad2d(10),
            nn.Conv2d(1, 1, 5, stride=1, padding=2, bias=False)
        )

        self.weights_init()
    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        n= np.zeros((5,5))
        n[3,3] = 1
        k = scipy.ndimage.gaussian_filter(n,sigma=1)
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))

class densenet:
    def __init__(self, pretrained=True, num_classes=2):
        self.pretrained = pretrained
        self.num_classes = num_classes

    def build_densenet(self):
        net = models.densenet.densenet161(pretrained=True)
        net.avgpool = nn.AdaptiveAvgPool2d(1)
        net.classifier = nn.Linear(2208, self.num_classes)
        return net

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # self.up1 = Up(2208, 1056 // 1, False)
        # self.conv1 = conv3x3(2208, 2112)
        # self.gn2_00 = nn.GroupNorm(4, 2112)
        # self.dconv1 = DoubleConv(2112, 1056)
        self.conv2 = conv3x3(1056, 768)
        self.gn2_0 = nn.GroupNorm(4, 768)

        self.up2 = Up(768, 384 // 1, False)
        self.up3 = Up(384, 192 // 1, False)
        self.up4 = Up(192, 96 // 1, False)
        self.upsample4_conv = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2d_2_1 = conv3x3(96, 96)
        self.gn2_1 = nn.GroupNorm(4, 96)

        self.upsample5_conv = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2d_2_2 = conv3x3(96, 96)
        self.gn2_2 = nn.GroupNorm(4, 96)
        self.conv2d_2_3 = conv3x3(96, 1)
        self.inst2_3 = nn.InstanceNorm2d(1)
        self.gaussian_blur = GaussianLayer()

        self.kernel = nn.Parameter(torch.tensor(
            [[[0.0, 0.0, 0.0], [0.0, 1.0, random.uniform(-0.1, -0.9)], [0.0, 0.0, 0.0]],
             [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, random.uniform(-0.1, -0.9)]],
             [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, random.uniform(-0.1, -0.0), 0.0]],
             [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [random.uniform(-0.1, -0.9), 0.0, 0.0]],
             [[0.0, 0.0, 0.0], [random.uniform(-0.1, -0.9), 1.0, 0.0], [0.0, 0.0, 0.0]],
             [[random.uniform(-0.1, -0.9), 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
             [[0.0, random.uniform(-0.1, -0.9), 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
             [[0.0, 0.0, random.uniform(-0.1, -0.9)], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], ],
        ).unsqueeze(1))

        self.nms_conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        with torch.no_grad():
            self.nms_conv.weight = self.kernel.float()

class Densenet_with_skip(nn.Module):
    def __init__(self, model):
        super(Densenet_with_skip, self).__init__()
        self.model = model
        self.decoder = Decoder()

        self.firstconv = nn.Sequential(*(list(self.model.features.children())[:4]))

        self.block1 = nn.Sequential(*(list(self.model.features.children())[4:6]))
        self.block2 = nn.Sequential(*(list(self.model.features.children())[6:8]))
        self.block3 = nn.Sequential(*(list(self.model.features.children())[8:10]))
        self.block4 = nn.Sequential(*(list(self.model.features.children())[10:12]))
    def forward_pred(self, image):
        pred_net = self.model(image)
        return pred_net

    def forward_decode(self, image):
        identity = image

        blur = torchvision.transforms.GaussianBlur((7,7))
        image = blur(image)
        image = self.firstconv(image)
        image1 = self.block1(image)
        image2 = self.block2(image1)
        image3 = self.block3(image2)
        # image4 = self.block4(image3)

        reconst2 = self.decoder.conv2(image3)
        reconst2 = self.decoder.gn2_0(reconst2)
        reconst2 = F.relu(reconst2)
        reconst2 = self.decoder.up2(reconst2, image2)
        reconst3 = self.decoder.up3(reconst2, image1)
        # reconst3 = self.decoder.up3(image2, image1)
        reconst4 = self.decoder.up4(reconst3, image)
        reconst = self.decoder.upsample4_conv(reconst4)
        reconst = self.decoder.conv2d_2_1(reconst)
        reconst = self.decoder.gn2_1(reconst)
        reconst = F.relu(reconst)
        reconst = self.decoder.upsample5_conv(reconst)
        reconst = self.decoder.conv2d_2_2(reconst)
        reconst = self.decoder.gn2_2(reconst)
        reconst = F.relu(reconst)
        reconst = self.decoder.conv2d_2_3(reconst)
        reconst = self.decoder.inst2_3(reconst)

        reconst = F.relu(reconst)

        # return reconst

        blurred = self.decoder.gaussian_blur(reconst)
        gradients = kornia.filters.spatial_gradient(blurred, normalized=False)
        # Unpack the edges
        gx = gradients[:, :, 0]
        gy = gradients[:, :, 1]

        angle = torch.atan2(gy, gx)

        # Radians to Degrees
        import math
        angle = 180.0 * angle / math.pi

        # Round angle to the nearest 45 degree
        angle = torch.round(angle / 45) * 45
        nms_magnitude = self.decoder.nms_conv(blurred)
        # nms_magnitude = F.conv2d(blurred, kernel.unsqueeze(1), padding=kernel.shape[-1]//2)

        # Non-maximal suppression
        # Get the indices for both directions
        positive_idx = (angle / 45) % 8
        positive_idx = positive_idx.long()

        negative_idx = ((angle / 45) + 4) % 8
        negative_idx = negative_idx.long()

        # Apply the non-maximum suppression to the different directions
        channel_select_filtered_positive = torch.gather(nms_magnitude, 1, positive_idx)
        channel_select_filtered_negative = torch.gather(nms_magnitude, 1, negative_idx)

        channel_select_filtered = torch.stack(
            [channel_select_filtered_positive, channel_select_filtered_negative], 1
        )

        # is_max = channel_select_filtered.min(dim=1)[0] > 0.0

        # magnitude = reconst * is_max

        thresh = nn.Threshold(0.0, 0.0)
        max_matrix = channel_select_filtered.min(dim=1)[0]
        max_matrix = thresh(max_matrix)
        magnitude = torch.mul(reconst, max_matrix)
        # magnitude = torchvision.transforms.functional.invert(magnitude)
        # magnitude = self.decoder.sharpen(magnitude)
        # magnitude = self.decoder.threshold(magnitude)
        return magnitude

    def forward(self, image):
        reconst = self.forward_decode(image)
        pred = self.forward_pred(image)
        return pred, reconst