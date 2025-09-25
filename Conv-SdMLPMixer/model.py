# from model.se_block import SEBlock
import math
import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce
import numpy as np
import torch
import copy
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
#######################MDmlp模块#################
class MLPLayer(nn.Module):
    def __init__(self, base_dim, dim, factor, forw_perm, **axes_lengths):
        super(MLPLayer, self).__init__()
        back_perm = forw_perm.split("->")[1] + " -> " + forw_perm.split("->")[0]
        # print(forw_perm,back_perm)
        self.norm = nn.LayerNorm(base_dim)

        self.forw_rearr = Rearrange(forw_perm, **axes_lengths)
        self.dense1 = nn.Linear(dim, int(dim * factor))
        self.dense2 = nn.Linear(int(dim * factor), dim)
        self.gelu = nn.GELU()
        self.back_rearr = Rearrange(back_perm, **axes_lengths)

    def forward(self, x):
        y = x
        # print('in',y.shape)#in torch.Size([2, 14, 14, 320])
        y = y.permute(0,2,3,1)
        y = self.norm(y)

        # print(y.shape)
        B,H,W,C = y.shape
        y = y.view(B, H*W, C).contiguous()
        # print(y.shape)
        y = self.forw_rearr(y)
        # print('forw',y.shape) #torch.Size([2, 320, 14, 14])
        y = self.dense1(y)
        y = self.gelu(y)
        y = self.dense2(y)
        y = self.gelu(y)
        # print(y.shape)#torch.Size([2, 320, 14, 14])
        y = self.back_rearr(y)
        # print('back', y.shape)#back torch.Size([2, 14, 14, 320])
        y = y.reshape(B,C,H,W)
        y = y + x
        return y
class MDBlock(nn.Module):
    def __init__(self, patch_num, base_dim, factor):
        super(MDBlock, self).__init__()
        h = w = patch_num
        self.hlayer = MLPLayer(base_dim, patch_num, factor, "b (h w) c -> b w c h", h=h, w=w)
        self.wlayer = MLPLayer(base_dim, patch_num, factor, 'b (h w) c -> b c h w', h=h, w=w)
        self.clayer = MLPLayer(base_dim, 256, factor, 'b (h w) c -> b h w c', h=h, w=w)
        # self.dlayer = MLPLayer(base_dim, base_dim, factor, 'b (h w) c d -> b h w c d', h=h, w=w)

    def forward(self, x):
        y = self.hlayer(x)
        y = self.wlayer(y)
        y = self.clayer(y)
        # y = self.dlayer(y)
        return y


###########################多分支CNN模块########################
def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    if in_channels == out_channels:
        groups = out_channels
        result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                            bias=False))
    else:
        result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                            bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False,
                 layer_scale_init_value=1e-6):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = out_channels
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.GELU()


        self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(
                num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   padding=padding_11, groups=groups)
            # print('RepVGG Block, identity = ', self.rbr_identity)
        # print(in_channels)
        if out_channels == in_channels and stride == 1:   #########倒置瓶颈层
            self.pwconv1 = nn.Linear(in_channels, 4 * in_channels)
            self.pwconv2 = nn.Linear(4 * in_channels, in_channels)
        else:
            self.pwconv1 = nn.Linear(2 * in_channels, 4 * in_channels)
            self.pwconv2 = nn.Linear(4 * in_channels, 2 * in_channels)

    def forward(self, inputs):
        # print(inputs.shape)
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))
        # i = 0
        if self.rbr_identity is None:  # 判断是否使用shortcut
            # i+=1

            id_out = 0
        else:
            # print('ident',inputs.shape)
            id_out = self.rbr_identity(inputs)
            # print('ident1',id_out.shape)
        # print(inputs.shape[1])
        # exit()
        if inputs.shape[1] == 3:   #stem层
            # print('1', inputs.shape) #torch.Size([2, 3, 224, 224])
            x = self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out
            # print('1', x.shape)#torch.Size([2, 64, 112, 112])
            x = self.se(x)
            # print('se',x.shape)#1 torch.Size([2, 64, 112, 112])
            x = self.nonlinearity(x)
            # print('2', x.shape)#2 torch.Size([2, 64, 112, 112])
            return x
        else:  ####后面的其余层
            # print('1', inputs.shape)#1 torch.Size([2, 64, 112, 112])
            x = self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out
            # print('2', x.shape)#1 torch.Size([2, 128, 56, 56])
            x = self.se(x)
            # print('se2', x.shape)#1 torch.Size([2, 128, 56, 56])
            x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
            # print('pw1qian',x.shape)
            x = self.pwconv1(x)
            # print('4', x.shape)
            x = self.nonlinearity(x)
            x = self.pwconv2(x)
            x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
            # print('pw2',x.shape)
            return x
        # return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))


class RepVGG(nn.Module):

    def __init__(self, num_blocks, num_classes=1000, width_multiplier=None, override_groups_map=None, deploy=False,
                 use_se=False):
        super(RepVGG, self).__init__()

        assert len(width_multiplier) == 4

        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        self.use_se = use_se

        assert 0 not in self.override_groups_map

        self.in_planes = min(64, int(64 * width_multiplier[0]))  # stage0的输出通道数=64

        self.stage0 = RepVGGBlock(in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1,
                                  deploy=self.deploy, use_se=self.use_se)

        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(128, num_blocks[0], stride=2)
        self.stage2 = self._make_stage(256, num_blocks[1], stride=2)
        self.mdmlp0 = MDBlock(4,256,4)
        self.mdmlp1 = MDBlock(8,256,4)
        self.mdmlp2 = MDBlock(16,256,4)
        self.mdmlp3 = MDBlock(24,256,4)
        self.mdmlp4 = MDBlock(28,256,4)
        self.stage3 = self._make_stage(512, num_blocks[2], stride=2)
        self.stage4 = self._make_stage(1024, num_blocks[3], stride=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(1024, num_classes)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # 设置每个stage内每层卷积的步长，如stage2：[2, 1, 1, 1, 1, 1]
        # print(strides)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, deploy=self.deploy,
                                      use_se=self.use_se))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)
    def _upsample_add(self, x, y):######上采样相加操作，特征融合
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear',align_corners=True) + y

    def forward(self, x):
        out = self.stage0(x)
        # print(out.shape)
        out = self.stage1(out)
        out = self.stage2(out)

        B,C,H,W = out.shape
        feature1 = out[:, :, :W-24, H-4:]##########重叠多尺度分块操作，默认分块数为5，特征图大小为 4*4 8*8 12*12 24*24 28*28
        feature2 = out[:, :, :W-20, H-8:]
        feature3 = out[:, :, :W-12, H-16:]
        feature4 = out[:, :, :W-4, H-24:]
        feature5 = out
        f1_out = self.mdmlp0(feature1)###############多维度MLP操作
        f2_out = self.mdmlp1(feature2)
        f3_out = self.mdmlp2(feature3)
        f4_out = self.mdmlp3(feature4)
        f5_out = self.mdmlp4(feature5)
        ########################特征融合操作
        out = out+self._upsample_add(self._upsample_add(self._upsample_add(self._upsample_add(f1_out, f2_out), f3_out), f4_out),f5_out)
        # out = out + self._upsample(f1_out, out) + self._upsample(f2_out, out) + self._upsample(f3_out,out) + self._upsample(f4_out, out)
        out = self.stage3(out)
        # B,C,H,W = out.shape
        # feature1 = out[:, :, :W-12, H-2:]
        # feature2 = out[:, :, :W-10, H-4:]
        # feature3 = out[:, :, :W-6, H-8:]
        # feature4 = out[:, :, :W-2, H-12:]
        # feature5 = out
        # f21_out = self.st_mlp3(feature1,2,2)
        # f22_out = self.st_mlp3(feature2,4,4)
        # f23_out = self.st_mlp3(feature3,8,8)
        # f24_out = self.st_mlp3(feature4,12,12)
        # f25_out = self.st_mlp3(feature5,14,14)
        # out = out+self._upsample_add(self._upsample_add(self._upsample_add(self._upsample_add(f21_out, f22_out), f23_out), f24_out),f25_out)
        out = self.stage4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
# print(g2_map)
g4_map = {l: 4 for l in optional_groupwise_layers}


# print(g4_map)

def create_RepVGG_A0(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy)


def create_RepVGG_A1(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy)


def create_RepVGG_A2(num_classes, deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=num_classes,
                  width_multiplier=[1.5, 1.5, 1.5, 2.75], override_groups_map=None, deploy=deploy)


def create_RepVGG_B0(num_classes, deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy)


def create_RepVGG_B1(num_classes, deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=None, deploy=deploy)


def create_RepVGG_B1g2(num_classes, deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g2_map, deploy=deploy)


def create_RepVGG_B1g4(num_classes, deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g4_map, deploy=deploy)


def create_RepVGG_B2(num_classes, deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy)


def create_RepVGG_B2g2(num_classes, deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g2_map, deploy=deploy)


def create_RepVGG_B2g4(num_classes, deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g4_map, deploy=deploy)


