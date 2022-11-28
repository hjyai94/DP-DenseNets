import torch
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn as nn
import pdb
import math
from ptflops import get_model_complexity_info

try:
    from DenseVoxNet import *
except:
    from model.DenseVoxNet import *

try:
    from Unet import *
except:
    from model.Unet import *
from model.MultiScaleUnet import *
from model.DeepMedic import DeepMedic

class ModelBuilder():
    def build_net(self, arch='FullyHierarchical'):

        if arch == 'FullyHierarchical':
            network = FullyHierarchical()
            return network
        elif arch == 'Unet':
            network = Unet()
            return network
        elif arch == 'DenseVoxNet':
            network = DenseVoxNet()
            return network
        elif arch == 'DeepMedic':
            network = DeepMedic()
            return network
        elif arch == 'MultiScaleUnet':
            network = MultiScaleUnet()
            return network
        else:
            raise Exception('Architecture undefined')

def DenseBlock(nin, nout, kernel_size=3, bias=True, padding=1, dilation=1):
    model = nn.Sequential(nn.BatchNorm3d(nin),
            nn.PReLU(),
            nn.Conv3d(nin, nout, kernel_size=kernel_size, bias=bias, padding=padding, dilation=dilation)
    )
    return model

class FullyHierarchical(nn.Module):
    def __init__(self):
        super(FullyHierarchical, self).__init__()

        # bottom-up pathway
        ## bu means bottom-up
        self.conv_bu_1 = nn.Conv3d(2, 24, kernel_size=3, padding=1)
        self.conv_bu_2 = DenseBlock(24, 12)
        self.conv_bu_3 = DenseBlock(36, 12)
        self.conv_bu_4 = DenseBlock(48, 12)
        self.conv_bu_5 = DenseBlock(60, 12)
        self.conv_bu_6 = DenseBlock(72, 12)
        self.conv_bu_7 = DenseBlock(84, 12)
        self.conv_bu_8 = DenseBlock(96, 12)
        self.conv_bu_9 = DenseBlock(108, 12)
        self.conv_bu_10 = DenseBlock(120, 12)
        self.conv_bu_11 = DenseBlock(132, 12)
        self.conv_bu_12 = DenseBlock(144, 12)
        self.conv_bu_13 = DenseBlock(156, 12)

        self.fully_bu_1 = nn.Conv3d(168, 400, kernel_size=1)
        self.fully_bu_2 = nn.Conv3d(400, 100, kernel_size=1)
        self.final_bu = nn.Conv3d(100, 5, kernel_size=1)

        # top-down pathway
        ## tp means top-down
        self.conv_td_1 = nn.Conv3d(2, 24, kernel_size=3, padding=1)
        self.conv_td_2 = DenseBlock(192, 12)
        self.conv_td_3 = DenseBlock(204, 12)
        self.conv_td_4 = DenseBlock(216, 12)
        self.conv_td_5 = DenseBlock(228, 12)
        self.conv_td_6 = DenseBlock(240, 12)
        self.conv_td_7 = DenseBlock(252, 12)
        self.conv_td_8 = DenseBlock(264, 12)
        self.conv_td_9 = DenseBlock(276, 12)
        self.conv_td_10 = DenseBlock(288, 12)
        self.conv_td_11 = DenseBlock(300, 12)
        self.conv_td_12 = DenseBlock(312, 12)
        self.conv_td_13 = DenseBlock(324, 12)

        self.fully_td_1 = nn.Conv3d(336, 400, kernel_size=1)
        self.fully_td_2 = nn.Conv3d(400, 100, kernel_size=1)
        self.final_td = nn.Conv3d(100, 5, kernel_size=1)

    def forward(self, input):
        # ----- Bottop-up Pathway ------ #
        # ----- First layer ------ #
        x = self.conv_bu_1(input[:, 0:2, :, :, :])
        concat_bu_1 = x

        # ----- Second layer ------ #
        x = self.conv_bu_2(x)
        x = torch.cat((concat_bu_1, x), dim=1)
        del concat_bu_1
        concat_bu_2 = x

        # -----Third layer ------ #
        x = self.conv_bu_3(x)
        x = torch.cat((concat_bu_2, x), dim=1)
        del concat_bu_2
        concat_bu_3 = x

        x = self.conv_bu_4(x)
        x = torch.cat((concat_bu_3, x), dim=1)
        del concat_bu_3
        concat_bu_4 = x

        x = self.conv_bu_5(x)
        x = torch.cat((concat_bu_4, x), dim=1)
        del concat_bu_4
        concat_bu_5 = x

        x = self.conv_bu_6(x)
        x = torch.cat((concat_bu_5, x), dim=1)
        del concat_bu_5
        concat_bu_6 = x

        x = self.conv_bu_7(x)
        x = torch.cat((concat_bu_6, x), dim=1)
        del concat_bu_6
        concat_bu_7 = x

        x = self.conv_bu_8(x)
        x = torch.cat((concat_bu_7, x), dim=1)
        del concat_bu_7
        concat_bu_8 = x

        x = self.conv_bu_9(x)
        x = torch.cat((concat_bu_8, x), dim=1)
        del concat_bu_8
        concat_bu_9 = x

        x = self.conv_bu_10(x)
        x = torch.cat((concat_bu_9, x), dim=1)
        del concat_bu_9
        concat_bu_10 = x

        x = self.conv_bu_11(x)
        x = torch.cat((concat_bu_10, x), dim=1)
        del concat_bu_10
        concat_bu_11 = x

        x = self.conv_bu_12(x)
        x = torch.cat((concat_bu_11, x), dim=1)
        del concat_bu_11
        concat_bu_12 = x

        x = self.conv_bu_13(x)
        x = torch.cat((concat_bu_12, x), dim=1)
        del concat_bu_12
        concat_bu_13 = x

        x = self.fully_bu_1(x)
        x = self.fully_bu_2(x)
        out_bu = self.final_bu(x)

        # ----- Top-down Pathway ------ #
        # ----- First layer ------ #
        x = self.conv_td_1(input[:, 2:4, :, :, :])
        x = torch.cat((concat_bu_13, x), dim=1)
        concat_td_1 = x

        # ----- Second layer ------ #
        x = self.conv_td_2(x)
        x = torch.cat((concat_td_1, x), dim=1)
        del concat_td_1
        concat_td_2 = x

        # -----Third layer ------ #
        x = self.conv_td_3(x)
        x = torch.cat((concat_td_2, x), dim=1)
        del concat_td_2
        concat_td_3 = x

        x = self.conv_td_4(x)
        x = torch.cat((concat_td_3, x), dim=1)
        del concat_td_3
        concat_td_4 = x

        x = self.conv_td_5(x)
        x = torch.cat((concat_td_4, x), dim=1)
        del concat_td_4
        concat_td_5 = x

        x = self.conv_td_6(x)
        x = torch.cat((concat_td_5, x), dim=1)
        del concat_td_5
        concat_td_6 = x

        x = self.conv_td_7(x)
        x = torch.cat((concat_td_6, x), dim=1)
        del concat_td_6
        concat_td_7 = x

        x = self.conv_td_8(x)
        x = torch.cat((concat_td_7, x), dim=1)
        del concat_td_7
        concat_td_8 = x

        x = self.conv_td_9(x)
        x = torch.cat((concat_td_8, x), dim=1)
        del concat_td_8
        concat_td_9 = x

        x = self.conv_td_10(x)
        x = torch.cat((concat_td_9, x), dim=1)
        del concat_td_9
        concat_td_10 = x

        x = self.conv_td_11(x)
        x = torch.cat((concat_td_10, x), dim=1)
        del concat_td_10
        concat_td_11 = x

        x = self.conv_td_12(x)
        x = torch.cat((concat_td_11, x), dim=1)
        del concat_td_11
        concat_td_12 = x

        x = self.conv_td_13(x)
        x = torch.cat((concat_td_12, x), dim=1)
        del concat_td_12

        x = self.fully_td_1(x)
        x = self.fully_td_2(x)
        out_td = self.final_td(x)

        return out_bu, out_td



if __name__ == '__main__':
    image = torch.randn(2, 4, 48,48, 48).cuda()
    print(image.shape)
    builder = ModelBuilder()
    model = builder.build_net('Unet').cuda()
    out = model(image)
    print(out.shape)

    # num_classes = 5
    # model = FullyHierarchical().cuda()
    # out_bu, out_td = model(image)
    # print(out_bu.shape, out_td.shape)
    # flops, params = get_model_complexity_info(model, (4, 38, 38, 38), as_strings=True, print_per_layer_stat=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

