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

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.up1 = Up(2048, 1024 // 1, False)
        self.up2 = Up(1024, 512 // 1, False)
        self.up3 = Up(512, 256 // 1, False)
        self.conv2d_2_1 = conv3x3(256, 128)
        self.gn1 = nn.GroupNorm(4, 128)
        self.instance1 = nn.InstanceNorm2d(128)
        self.up4 = Up(128, 64 // 1, False)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.upsample4 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.upsample4_conv = DoubleConv(64, 64, 64 // 2)
        self.up_ = Up(128, 128 // 1, False)
        self.conv2d_2_2 = conv3x3(128, 6)
        self.instance2 = nn.InstanceNorm2d(6)
        self.gn2 = nn.GroupNorm(3, 6)
        self.gaussian_blur = GaussianLayer()
        self.up5 = Up(6, 3, False)
        self.conv2d_2_3 = conv3x3(3, 1)
        self.instance3 = nn.InstanceNorm2d(1)
        self.gaussian_blur = GaussianLayer()
        self.kernel = nn.Parameter(torch.tensor(
            [[[0.0, 0.0, 0.0], [0.0, 1.0, random.uniform(-1.0, 0.0)], [0.0, 0.0, 0.0]],
             [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, random.uniform(-1.0, 0.0)]],
             [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, random.uniform(random.uniform(-1.0, 0.0), -0.0), 0.0]],
             [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [random.uniform(-1.0, 0.0), 0.0, 0.0]],
             [[0.0, 0.0, 0.0], [random.uniform(-1.0, 0.0), 1.0, 0.0], [0.0, 0.0, 0.0]],
             [[random.uniform(-1.0, 0.0), 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
             [[0.0, random.uniform(-1.0, 0.0), 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
             [[0.0, 0.0, random.uniform(-1.0, 0.0)], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], ],
        ).unsqueeze(1))

        self.nms_conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        with torch.no_grad():
            self.nms_conv.weight = self.kernel.float()


class Resnet_with_skip(nn.Module):
    def __init__(self, model):
        super(Resnet_with_skip, self).__init__()
        self.model = model
        self.decoder = Decoder()

    def forward_pred(self, image):
        pred_net = self.model(image)
        return pred_net

    def forward_decode(self, image):
        identity = image

        image = self.model.conv1(image)
        image = self.model.bn1(image)
        image = self.model.relu(image)
        image1 = self.model.maxpool(image)

        image2 = self.model.layer1(image1)
        image3 = self.model.layer2(image2)
        image4 = self.model.layer3(image3)
        image5 = self.model.layer4(image4)

        reconst1 = self.decoder.up1(image5, image4)
        reconst2 = self.decoder.up2(reconst1, image3)
        reconst3 = self.decoder.up3(reconst2, image2)
        reconst = self.decoder.conv2d_2_1(reconst3)
        # reconst = self.decoder.instance1(reconst)
        reconst = self.decoder.gn1(reconst)
        reconst = F.relu(reconst)
        reconst4 = self.decoder.up4(reconst, image1)
        # reconst5 = self.decoder.upsample4(reconst4)
        reconst5 = self.decoder.upsample4(reconst4)
        # reconst5 = self.decoder.upsample4_conv(reconst4)
        reconst5 = self.decoder.up_(reconst5, image)
        # reconst5 = reconst5 + image
        reconst5 = self.decoder.conv2d_2_2(reconst5)
        reconst5 = self.decoder.instance2(reconst5)
        # reconst5 = self.decoder.gn2(reconst5)
        reconst5 = F.relu(reconst5)
        reconst = self.decoder.up5(reconst5, identity)
        reconst = self.decoder.conv2d_2_3(reconst)
        # reconst = self.decoder.instance3(reconst)
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

        thresh = nn.Threshold(0.01, 0.01)
        max_matrix = channel_select_filtered.min(dim=1)[0]
        max_matrix = thresh(max_matrix)
        magnitude = torch.mul(reconst, max_matrix)
        # magnitude = torchvision.transforms.functional.invert(magnitude)
        # magnitude = self.decoder.sharpen(magnitude)
        # magnitude = self.decoder.threshold(magnitude)
        magnitude = kornia.enhance.adjust_gamma(magnitude, 2.0)
        # magnitude = F.leaky_relu(magnitude)
        return magnitude

    def forward(self, image):
        reconst = self.forward_decode(image)
        pred = self.forward_pred(image)
        return pred, reconst
