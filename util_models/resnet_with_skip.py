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

# class Decoder(nn.Module):
#     def __init__(self):
#         super(Decoder, self).__init__()
#         self.up1 = Up(2048, 1024 // 1, False)
#         self.up2 = Up(1024, 512 // 1, False)
#         self.up3 = Up(512, 256 // 1, False)
#         self.upsample3_conv = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.conv2d_2_1 = conv3x3(256, 128)
#         self.gn1 = nn.GroupNorm(4, 128)
#         self.up4 = Up(128, 64 // 1, False)
#         self.upsample4_conv = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.conv2d_2_2 = conv3x3(64, 64)
#         self.gn2 = nn.GroupNorm(4, 64)
#         self.conv2d_2_3 = conv3x3(64, 1)
#         self.gaussian_blur = GaussianLayer()
#         self.instance1 = nn.InstanceNorm2d(1)
#         self.kernel = nn.Parameter(torch.tensor(
#             [[[0.0, 0.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 0.0]],
#              [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
#              [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, random.uniform(-0.1, -0.0), 0.0]],
#              [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
#              [[0.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
#              [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
#              [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
#              [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], ],
#         ).unsqueeze(1))
#
#         self.nms_conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
#         with torch.no_grad():
#             self.nms_conv.weight = self.kernel.float()
#
# class Resnet_with_skip(nn.Module):
#     def __init__(self, model):
#         super(Resnet_with_skip, self).__init__()
#         self.model = model
#         self.decoder = Decoder()
#
#     def forward_pred(self, image):
#         pred_net = self.model(image)
#         return pred_net
#
#     def forward_decode(self, image):
#         identity = image
#
#         image = self.model.conv1(image)
#         image1 = self.model.bn1(image)
#         image1 = self.model.relu(image1)
#         image1 = self.model.maxpool(image1)
#
#         image1 = self.model.layer1(image1)
#         image2 = self.model.layer2(image1)
#         image3 = self.model.layer3(image2)
#         image4 = self.model.layer4(image3)
#
#         reconst1 = self.decoder.up1(image4, image3)
#         reconst2 = self.decoder.up2(reconst1, image2)
#         reconst3 = self.decoder.up3(reconst2, image1)
#         reconst = self.decoder.conv2d_2_1(reconst3)
#         reconst = self.decoder.gn1(reconst)
#         reconst = F.leaky_relu(reconst)
#         reconst = self.decoder.upsample3_conv(reconst)
#         reconst4 = self.decoder.up4(reconst, image)
#         reconst = self.decoder.upsample4_conv(reconst4)
#         # reconst = self.decoder.conv2d_2_2(reconst)
#         # reconst = self.decoder.gn2(reconst)
#         # reconst = F.leaky_relu(reconst)
#         reconst = self.decoder.conv2d_2_3(reconst)
#         reconst = self.decoder.instance1(reconst)
#         # reconst = torchvision.transforms.functional.invert(reconst)
#         # return reconst
#
#         blurred = self.decoder.gaussian_blur(reconst)
#         gradients = kornia.filters.spatial_gradient(blurred, normalized=False)
#         # Unpack the edges
#         gx = gradients[:, :, 0]
#         gy = gradients[:, :, 1]
#
#         angle = torch.atan2(gy, gx)
#
#         # Radians to Degrees
#         import math
#         angle = 180.0 * angle / math.pi
#
#         # Round angle to the nearest 45 degree
#         angle = torch.round(angle / 45) * 45
#         nms_magnitude = self.decoder.nms_conv(blurred)
#         # nms_magnitude = F.conv2d(blurred, kernel.unsqueeze(1), padding=kernel.shape[-1]//2)
#
#         # Non-maximal suppression
#         # Get the indices for both directions
#         positive_idx = (angle / 45) % 8
#         positive_idx = positive_idx.long()
#
#         negative_idx = ((angle / 45) + 4) % 8
#         negative_idx = negative_idx.long()
#
#         # Apply the non-maximum suppression to the different directions
#         channel_select_filtered_positive = torch.gather(nms_magnitude, 1, positive_idx)
#         channel_select_filtered_negative = torch.gather(nms_magnitude, 1, negative_idx)
#
#         channel_select_filtered = torch.stack(
#             [channel_select_filtered_positive, channel_select_filtered_negative], 1
#         )
#
#         # is_max = channel_select_filtered.min(dim=1)[0] > 0.0
#
#         # magnitude = reconst * is_max
#
#         thresh = nn.Threshold(0.0, 0.0)
#         max_matrix = channel_select_filtered.min(dim=1)[0]
#         max_matrix = thresh(max_matrix)
#         magnitude = torch.mul(reconst, max_matrix)
#         # magnitude = torchvision.transforms.functional.invert(magnitude)
#         # magnitude = self.decoder.sharpen(magnitude)
#         # magnitude = self.decoder.threshold(magnitude)
#         magnitude = kornia.enhance.adjust_gamma(magnitude, 4.0)
#         magnitude = F.leaky_relu(magnitude)
#         return magnitude
#
#     def forward(self, image):
#         reconst = self.forward_decode(image)
#         pred = self.forward_pred(image)
#         return pred, reconst

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.up1 = Up(2048, 1024 // 1, False)
        self.up2 = Up(1024, 512 // 1, False)
        self.up3 = Up(512, 256 // 1, False)
        self.upsample3_conv = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2d_2_1 = conv3x3(256, 128)
        self.gn1 = nn.GroupNorm(4, 128)
        self.instance1 = nn.InstanceNorm2d(128)
        self.up4 = Up(128, 64 // 1, False)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2d_2_2 = conv3x3(64, 6)
        self.instance2 = nn.InstanceNorm2d(6)
        self.gaussian_blur = GaussianLayer()
        self.instance2 = nn.InstanceNorm2d(1)
        self.up5 = Up(6, 3, False)
        self.conv2d_2_3 = conv3x3(3, 1)
        self.instance3 = nn.InstanceNorm2d(1)
        self.gaussian_blur = GaussianLayer()
        self.kernel = nn.Parameter(torch.tensor(
            [[[0.0, 0.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 0.0]],
             [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
             [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, random.uniform(-0.1, -0.0), 0.0]],
             [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
             [[0.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
             [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
             [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
             [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], ],
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
        reconst = self.decoder.instance1(reconst)
        reconst = F.relu(reconst)
        # reconst = self.decoder.upsample3_conv(reconst)
        reconst4 = self.decoder.up4(reconst, image1)
        reconst5 = self.decoder.upsample4(reconst4)
        reconst5 = reconst5 + image
        reconst5 = self.decoder.conv2d_2_2(reconst5)
        reconst5 = self.decoder.instance2(reconst5)
        reconst5 = F.relu(reconst5)
        reconst = self.decoder.up5(reconst5, identity)
        reconst = self.decoder.conv2d_2_3(reconst)
        reconst = self.decoder.instance3(reconst)
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
        magnitude = kornia.enhance.adjust_gamma(magnitude, 4.0)
        magnitude = F.leaky_relu(magnitude)
        return magnitude

    def forward(self, image):
        reconst = self.forward_decode(image)
        pred = self.forward_pred(image)
        return pred, reconst


# class _DenseLayer(nn.Sequential):
#     def __init__(self, input_features, out_features):
#         super(_DenseLayer, self).__init__()
#
#         # self.add_module('relu2', nn.ReLU(inplace=True)),
#         self.add_module('conv1', nn.Conv2d(input_features, out_features,
#                                            kernel_size=3, stride=1, padding=2, bias=True)),
#         self.add_module('norm1', nn.BatchNorm2d(out_features)),
#         self.add_module('relu1', nn.ReLU(inplace=True)),
#         self.add_module('conv2', nn.Conv2d(out_features, out_features,
#                                            kernel_size=3, stride=1, bias=True)),
#         self.add_module('norm2', nn.BatchNorm2d(out_features))
#
#     def forward(self, x):
#         x1, x2 = x
#
#         new_features = super(_DenseLayer, self).forward(F.relu(x1))  # F.relu()
#         # if new_features.shape[-1]!=x2.shape[-1]:
#         #     new_features =F.interpolate(new_features,size=(x2.shape[2],x2.shape[-1]), mode='bicubic',
#         #                                 align_corners=False)
#         return 0.5 * (new_features + x2), x2
#
#
# class _DenseBlock(nn.Sequential):
#     def __init__(self, num_layers, input_features, out_features):
#         super(_DenseBlock, self).__init__()
#         for i in range(num_layers):
#             layer = _DenseLayer(input_features, out_features)
#             self.add_module('denselayer%d' % (i + 1), layer)
#             input_features = out_features
#
#
# class UpConvBlock(nn.Module):
#     def __init__(self, in_features, up_scale):
#         super(UpConvBlock, self).__init__()
#         self.up_factor = 2
#         self.constant_features = 16
#
#         layers = self.make_deconv_layers(in_features, up_scale)
#         assert layers is not None, layers
#         self.features = nn.Sequential(*layers)
#
#     def make_deconv_layers(self, in_features, up_scale):
#         layers = []
#         all_pads=[0,0,1,3,7]
#         for i in range(up_scale):
#             kernel_size = 2 ** up_scale
#             pad = all_pads[up_scale]  # kernel_size-1
#             out_features = self.compute_out_features(i, up_scale)
#             layers.append(nn.Conv2d(in_features, out_features, 1))
#             layers.append(nn.ReLU(inplace=True))
#             layers.append(nn.ConvTranspose2d(
#                 out_features, out_features, kernel_size, stride=2, padding=pad))
#             in_features = out_features
#         return layers
#
#     def compute_out_features(self, idx, up_scale):
#         return 1 if idx == up_scale - 1 else self.constant_features
#
#     def forward(self, x):
#         return self.features(x)
#
#
# class SingleConvBlock(nn.Module):
#     def __init__(self, in_features, out_features, stride,
#                  use_bs=True
#                  ):
#         super(SingleConvBlock, self).__init__()
#         self.use_bn = use_bs
#         self.conv = nn.Conv2d(in_features, out_features, 1, stride=stride,
#                               bias=True)
#         self.bn = nn.BatchNorm2d(out_features)
#
#     def forward(self, x):
#         x = self.conv(x)
#         if self.use_bn:
#             x = self.bn(x)
#         return x
#
#
# class DoubleConvBlock(nn.Module):
#     def __init__(self, in_features, mid_features,
#                  out_features=None,
#                  stride=1,
#                  use_act=True):
#         super(DoubleConvBlock, self).__init__()
#
#         self.use_act = use_act
#         if out_features is None:
#             out_features = mid_features
#         self.conv1 = nn.Conv2d(in_features, mid_features,
#                                3, padding=1, stride=stride)
#         self.bn1 = nn.BatchNorm2d(mid_features)
#         self.conv2 = nn.Conv2d(mid_features, out_features, 3, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_features)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         if self.use_act:
#             x = self.relu(x)
#         return x
#
# class Resnet_with_skip(nn.Module):
#     def __init__(self, model):
#         super(Resnet_with_skip, self).__init__()
#         self.model = model
#
#         self.block_1 = DoubleConvBlock(3, 32, 64, stride=2,)
#         self.block_2 = DoubleConvBlock(64, 128, use_act=False)
#         self.dblock_3 = _DenseBlock(2, 128, 256) # [128,256,100,100]
#         self.dblock_4 = _DenseBlock(3, 256, 512)
#         self.dblock_5 = _DenseBlock(3, 512, 512)
#         self.dblock_6 = _DenseBlock(3, 512, 256)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#         # left skip connections, figure in Journal
#         self.side_0 = SingleConvBlock(64, 256, 2)
#         self.side_1 = SingleConvBlock(256, 512, 2)
#         self.side_2 = SingleConvBlock(512, 1024, 2)
#         self.side_3 = SingleConvBlock(1024, 2048, 1)
#         self.side_4 = SingleConvBlock(2048, 2048, 1)
#
#         self.last_conv1 = SingleConvBlock(2048, 1024, 1)
#         self.last_conv2 = SingleConvBlock(1024, 512, 1)
#         self.last_conv3 = SingleConvBlock(512, 256, 1)
#         self.last_conv4 = SingleConvBlock(256, 64, 1)
#         self.last_conv5 = SingleConvBlock(64, 1, 1)
#
#         self.up_block_0 = UpConvBlock(256, 3)
#         self.up_block_1 = UpConvBlock(512, 4)
#         self.up_block_2 = UpConvBlock(1024, 4)
#         self.up_block_3 = UpConvBlock(2048, 4)
#         self.up_block_4 = UpConvBlock(2048, 4)
#         self.up_block_5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.up_block_6 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
#         self.block_cat = SingleConvBlock(5, 1, stride=1, use_bs=False) # hed fusion method
#
#     def forward_pred(self, image):
#         pred_net = self.model(image)
#         return pred_net
#
#     def forward_decode(self, image):
#         identity = image
#
#         image = self.model.conv1(image)
#         image0 = self.model.bn1(image)
#         image0 = self.model.relu(image0)
#         image0 = self.model.maxpool(image0)
#
#         image0_side = self.side_0(image0)
#
#         image1 = self.model.layer1(image0)
#         image1_down = self.maxpool(image1)
#         image1_add = image1_down + image0_side
#         image1_side = self.side_1(image1_add)
#
#         image2 = self.model.layer2(image1)
#         image2_down = self.maxpool(image2)
#         image2_add = image2_down + image1_side
#         image2_side = self.side_2(image2_add)
#
#         image3 = self.model.layer3(image2)
#         image3_down = self.maxpool(image3)
#         image3_add = image3_down + image2_side
#         image3_side = self.side_3(image3_add)
#
#         image4 = self.model.layer4(image3)
#         # image4_down = self.maxpool(image4)
#         image4_add = image4 + image3_side
#         image4_side = self.side_4(image4_add)
#
#         # image4_last = self.last_conv1(image4_side)
#         # image4_last = self.last_conv2(image4_last)
#         # image4_last = self.last_conv3(image4_last)
#         # image4_last = self.last_conv4(image4_last)
#         # image4_last = self.last_conv5(image4_last)
#
#         # upsampling blocks
#         out_0 = self.up_block_0(image0_side)
#         out_1 = self.up_block_1(image1_side)
#         out_2 = self.up_block_2(image2_side)
#         out_2 = self.up_block_5(out_2)
#         out_3 = self.up_block_3(image3_side)
#         out_3 = self.up_block_5(out_3)
#         out_4 = self.up_block_4(image4_side)
#         out_4 = self.up_block_5(out_4)
#
#         results = [out_0, out_1, out_2, out_3, out_4]
#
#         block_cat = torch.cat(results, dim=1)
#         block_cat = self.block_cat(block_cat)
#
#         return out_0
#
#     def forward(self, image):
#         reconst = self.forward_decode(image)
#         pred = self.forward_pred(image)
#         return pred, reconst