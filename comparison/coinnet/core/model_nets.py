import os
import torch

from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from comparison.coinnet.core import resnet, densenet

cbp_in  = 2208
cbp_out = 16000

class CompactBilinearPooling(torch.nn.Module):
    def __init__(self, input_dim1, input_dim2, output_dim, sum_pool = True):
        super(CompactBilinearPooling, self).__init__()
        self.output_dim = output_dim
        self.sum_pool = sum_pool
        generate_sketch_matrix = lambda rand_h, rand_s, input_dim, output_dim: torch.sparse.FloatTensor(torch.stack([torch.arange(input_dim, out = torch.LongTensor()), rand_h.long()]), rand_s.float(), [input_dim, output_dim]).to_dense()
        self.sketch_matrix1 = torch.nn.Parameter(generate_sketch_matrix(torch.randint(output_dim, size = (input_dim1,)), 2 * torch.randint(2, size = (input_dim1,)) - 1, input_dim1, output_dim))
        self.sketch_matrix2 = torch.nn.Parameter(generate_sketch_matrix(torch.randint(output_dim, size = (input_dim2,)), 2 * torch.randint(2, size = (input_dim2,)) - 1, input_dim2, output_dim))

    def forward(self, x1, x2):

        fft1 = torch.fft.rfft(x1.permute(0, 2, 3, 1).matmul(self.sketch_matrix1), 1).squeeze()
        fft2 = torch.fft.rfft(x2.permute(0, 2, 3, 1).matmul(self.sketch_matrix2), 1).squeeze()

        fft_product = torch.stack([fft1[..., 0] * fft2[..., 0] - fft1[..., 1] * fft2[..., 1], fft1[..., 0] * fft2[..., 1] + fft1[..., 1] * fft2[..., 0]], dim = -1)
        output_dim = self.output_dim
        cbp = torch.fft.irfft(fft_product, 1) * self.output_dim

        # return cbp.sum(dim = 1).sum(dim = 1) if self.sum_pool else cbp.permute(0, 3, 1, 2)
        return cbp


class netBlock(nn.Module):
    def __init__(self,cbp_out):
        super(netBlock, self).__init__()

        self.conv_b1 = BasicBlock(cbp_out, 512)
        self.conv_r1 = ResidualBlock(512, 512)
        self.conv_r2 = ResidualBlock(512, 512)
        self.conv_r3 = ResidualBlock(512, 512)
        self.conv_b2 = BasicBlock(512, 1)

    def forward(self, x):      
        conv_b1 = self.conv_b1(x)
        conv_r1 = self.conv_r1(conv_b1)
        conv_r2 = self.conv_r2(conv_r1)
        conv_r3 = self.conv_r3(conv_r2)
        conv_b2 = self.conv_b2(conv_r3)
        return conv_b2

class ResidualBlock(nn.Module):
    def __init__(self, 
                 in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

        
    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, pad=1):
        super(BasicBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True)
        )
      
    def forward(self, x):
        out = self.body(x)
        return out


class model_r50_net(nn.Module):
    def __init__(self):
        super(model_r50_net, self).__init__()

        self.model_d161 = densenet.densenet161(pretrained=True)
        self.model_d161.avgpool = nn.AdaptiveAvgPool2d(1)
        self.model_d161.classifier = nn.Linear(cbp_in, 100)


        self.fc = nn.Linear(cbp_in, 100)

        self.cbp_layer_feat = CompactBilinearPooling(cbp_in, cbp_in, cbp_out, sum_pool = False)


        self.conv_block = netBlock(cbp_out)          
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):

        #_, r50_feat,  _ = self.model_r50(x)
        _, d161_features, _ = self.model_d161(x)
        #d161_feat = self.d161_conv_(d161_features)

        cbp_feat = self.cbp_layer_feat(d161_features, d161_features) 
        conv_blk = self.conv_block(cbp_feat)
        f_out = F.softmax(conv_blk, dim=1) + d161_features

        f_avg_out    = self.avg_pool(f_out)

        #print(f_avg_out.shape)
        #print(d161_avg_out.shape)

        #cbp_feat_fc  = self.cbp_layer_fc(f_avg_out, d161_avg_out) 
        cbp_feat_v = f_avg_out.view(f_avg_out.size(0), -1)
        fc_out  = self.fc(cbp_feat_v)

        return fc_out


