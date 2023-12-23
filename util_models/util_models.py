import torch.nn.utils.prune

from utils_functions.imports import *

from comparison.glyphnet.models.model import *

from comparison.coinnet.core.model_nets import model_r50_net



class resnet:
    def __init__(self, arch_type, args, pretrained=False, num_classes=2):
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.arch_type = arch_type
        self.arch_type = arch_type
        self.parameters_args = args

    def build_resnet(self):
        if self.arch_type=="resnet152":
            resnet = models.resnet152(pretrained=self.pretrained) # resnet152
        elif self.arch_type=="resnet101":
            resnet = models.resnet101(pretrained=self.pretrained)  # resnet101
        elif self.arch_type=="resnet50":
            resnet = models.resnet50(pretrained=self.pretrained)  # resnet50
        num_ftrs = resnet.fc.in_features
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        resnet.fc = nn.Linear(num_ftrs, self.num_classes)
        return resnet

def get_model(arch_type, num_classes, args):
    if "resnet" in arch_type:
        pretrained = False
        if "pretrained" in arch_type:
            arch_type = arch_type.split("_")[1]
            pretrained = True
        model = resnet(arch_type, args, pretrained=pretrained, num_classes=num_classes).build_resnet()
        return model
    elif "glyphnet" in arch_type:
        dropout = 0.15
        first_convolution_filters = 64
        last_separable_convolution_filters = 512
        inner_separable_convolution_filters_seq = [128, 128, 256, 256]
        return Glyphnet(num_classes=num_classes,
                     first_conv_out=first_convolution_filters,
                     last_sconv_out=last_separable_convolution_filters,
                     sconv_seq_outs=inner_separable_convolution_filters_seq,
                     dropout_rate=dropout
                     )
    elif "coinnet" in arch_type:
        net_ = model_r50_net()
        cbp_in = 2208
        cbp_out = 16000
        if "pretrained" in arch_type:
            ckpt = torch.load('comparison/coinnet/models/best_model.ckpt')
            net_.load_state_dict(ckpt['net_state_dict'])
        net_.model_d161.classifier = nn.Linear(cbp_in, num_classes)


        net_.fc = nn.Linear(cbp_in, num_classes)
        net_ = torch.nn.DataParallel(net_)
        return net_
    elif "densenet" in arch_type:
        net = models.densenet.densenet161(pretrained=True)
        net.avgpool = nn.AdaptiveAvgPool2d(1)
        net.classifier = nn.Linear(2208, num_classes)
        return net
    elif "efficientnet" in arch_type:
        model = models.efficientnet_b3(pretrained=True)
        model.classifier[1] = nn.Linear(in_features=1536, out_features=num_classes)
        return model