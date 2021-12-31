"""
IndexNet Matting
REF: https://github.com/poppinace/indexnet_matting/blob/ddb374a3e0e1ef3042437b7a88785950d6b0fbe1/scripts/hlindex.py#L37

Indices Matter: Learning to Index for Deep Image Matting
IEEE/CVF International Conference on Computer Vision, 2019
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..bn_helper import BatchNorm2d_class


class HolisticIndexBlock(nn.Module):
    def __init__(self, inp, use_nonlinear=False, use_context=False, batch_norm=nn.BatchNorm2d):
        super(HolisticIndexBlock, self).__init__()

        if use_context:
            kernel_size, padding = 4, 1
        else:
            kernel_size, padding = 2, 0

        if use_nonlinear:
            self.indexnet = nn.Sequential(
                nn.Conv2d(inp, 2 * inp, kernel_size=kernel_size, stride=2, padding=padding, bias=False),
                batch_norm(2 * inp),
                nn.ReLU6(inplace=True),
                nn.Conv2d(2 * inp, 4, kernel_size=1, stride=1, padding=0, bias=False)
            )
        else:
            self.indexnet = nn.Conv2d(inp, 4, kernel_size=kernel_size, stride=2, padding=padding, bias=False)

    def forward(self, x):
        x = self.indexnet(x)

        y = torch.sigmoid(x)
        z = F.softmax(y, dim=1)

        idx_en = F.pixel_shuffle(z, 2)
        idx_de = F.pixel_shuffle(y, 2)

        return idx_en, idx_de


class DepthwiseO2OIndexBlock(nn.Module):
    def __init__(self, inp, use_nonlinear=False, use_context=False, batch_norm=nn.BatchNorm2d):
        super(DepthwiseO2OIndexBlock, self).__init__()

        self.indexnet1 = self._build_index_block(inp, use_nonlinear, use_context, batch_norm)
        self.indexnet2 = self._build_index_block(inp, use_nonlinear, use_context, batch_norm)
        self.indexnet3 = self._build_index_block(inp, use_nonlinear, use_context, batch_norm)
        self.indexnet4 = self._build_index_block(inp, use_nonlinear, use_context, batch_norm)

    def _build_index_block(self, inp, use_nonlinear, use_context, batch_norm):

        if use_context:
            kernel_size, padding = 4, 1
        else:
            kernel_size, padding = 2, 0

        if use_nonlinear:
            return nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=2, padding=padding, groups=inp, bias=False),
                batch_norm(inp),
                nn.ReLU6(inplace=True),
                nn.Conv2d(inp, inp, kernel_size=1, stride=1, padding=0, groups=inp, bias=False)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=2, padding=padding, groups=inp, bias=False)
            )

    def forward(self, x):
        bs, c, h, w = x.size()

        x1 = self.indexnet1(x).unsqueeze(2)
        x2 = self.indexnet2(x).unsqueeze(2)
        x3 = self.indexnet3(x).unsqueeze(2)
        x4 = self.indexnet4(x).unsqueeze(2)

        x = torch.cat((x1, x2, x3, x4), dim=2)

        # normalization
        y = torch.sigmoid(x)
        z = F.softmax(y, dim=2)
        # pixel shuffling
        y = y.view(bs, c * 4, int(h / 2), int(w / 2))
        z = z.view(bs, c * 4, int(h / 2), int(w / 2))
        idx_en = F.pixel_shuffle(z, 2)
        idx_de = F.pixel_shuffle(y, 2)

        return idx_en, idx_de


class DepthwiseM2OIndexBlock(nn.Module):
    def __init__(self, inp, use_nonlinear=False, use_context=False, batch_norm=nn.BatchNorm2d):
        super(DepthwiseM2OIndexBlock, self).__init__()
        self.use_nonlinear = use_nonlinear

        self.indexnet1 = self._build_index_block(inp, use_nonlinear, use_context, batch_norm)
        self.indexnet2 = self._build_index_block(inp, use_nonlinear, use_context, batch_norm)
        self.indexnet3 = self._build_index_block(inp, use_nonlinear, use_context, batch_norm)
        self.indexnet4 = self._build_index_block(inp, use_nonlinear, use_context, batch_norm)

    def _build_index_block(self, inp, use_nonlinear, use_context, batch_norm):

        if use_context:
            kernel_size, padding = 4, 1
        else:
            kernel_size, padding = 2, 0

        if use_nonlinear:
            return nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=2, padding=padding, bias=False),
                batch_norm(inp),
                nn.ReLU6(inplace=True),
                nn.Conv2d(inp, inp, kernel_size=1, stride=1, padding=0, bias=False)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=2, padding=padding, bias=False)
            )

    def forward(self, x):
        bs, c, h, w = x.size()

        x1 = self.indexnet1(x).unsqueeze(2)
        x2 = self.indexnet2(x).unsqueeze(2)
        x3 = self.indexnet3(x).unsqueeze(2)
        x4 = self.indexnet4(x).unsqueeze(2)

        x = torch.cat((x1, x2, x3, x4), dim=2)

        # normalization
        y = torch.sigmoid(x)
        z = F.softmax(y, dim=2)
        # pixel shuffling
        y = y.view(bs, c * 4, int(h / 2), int(w / 2))
        z = z.view(bs, c * 4, int(h / 2), int(w / 2))
        idx_en = F.pixel_shuffle(z, 2)
        idx_de = F.pixel_shuffle(y, 2)

        return idx_en, idx_de


class IndexDown(HolisticIndexBlock):
    def forward(self, x):
        x_new = self.indexnet(x)

        y = torch.sigmoid(x_new)
        z = F.softmax(y, dim=1)

        idx_en = F.pixel_shuffle(z, 2)
        idx_de = F.pixel_shuffle(y, 2)

        x = x * idx_en
        x = 4 * F.avg_pool2d(x, (2, 2), stride=2)
        return x, idx_de


class IndexDownModule(nn.Module):
    def __init__(self,
                 inplane, outplane, num=1,
                 kernel_size=3,
                 use_nonlinear=False,
                 use_context=False,
                 batch_norm=nn.BatchNorm2d):
        super(IndexDownModule, self).__init__()

        self.kernel_size = kernel_size
        self.num = num

        downsample_list = []
        for i in range(self.num):
            downsample_list.append(
                IndexDown(inplane, use_nonlinear=use_nonlinear, use_context=use_context, batch_norm=batch_norm))
        downsample_list.append(
            nn.Sequential(
                nn.Conv2d(
                    inplane,
                    outplane,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=self.kernel_size // 2,
                    bias=False),
                batch_norm(outplane),
            )
        )
        self.downsample_list = nn.Sequential(*downsample_list)

    def forward(self, x):
        index_list = []
        for i in range(self.num):
            x, index = self.downsample_list[i](x)
            index_list.append(index)
        x = self.downsample_list[-1](x)  # 1x1 conv
        return x, index_list


class IndexUpModule(nn.Module):
    def __init__(self,
                 inplane, outplane, num=1,
                 kernel_size=3,
                 batch_norm=nn.BatchNorm2d):
        super(IndexUpModule, self).__init__()
        self.kernel_size = kernel_size
        self.num = num

        self.conv = nn.Sequential(
            nn.Conv2d(
                inplane,
                outplane,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                bias=False),
            batch_norm(outplane)
        )

    def forward(self, x, index_list=None):
        x = self.conv(x)  # 1x1 conv
        for i in range(self.num):
            if index_list is not None:
                x = index_list[self.num - i - 1] * F.interpolate(x, scale_factor=2, mode='nearest')
            else:
                x = F.interpolate(x, scale_factor=2, mode='nearest')
        return x


class IndexedDecoder(nn.Module):
    def __init__(self, inp, oup, kernel_size=5, batch_norm=nn.BatchNorm2d):
        super(IndexedDecoder, self).__init__()

        self.upsample = nn.MaxUnpool2d((2, 2), stride=2)
        self.dconv = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False),
            batch_norm(oup),
            nn.ReLU6(inplace=True)
        )
        self._init_weight()

    def forward(self, l_encode, l_low, indices=None):
        l_encode = self.upsample(l_encode, indices) if indices is not None else l_encode
        l_encode = torch.cat((l_encode, l_low), dim=1)
        return self.dconv(l_encode)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, BatchNorm2d_class):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class IndexedUpsamlping(nn.Module):
    def __init__(self, inp, oup, kernel_size=5, batch_norm=nn.BatchNorm2d):
        super(IndexedUpsamlping, self).__init__()
        self.oup = oup

        self.dconv = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False),
            batch_norm(oup),
            nn.ReLU6(inplace=True)
        )
        self._init_weight()

    def forward(self, l_encode, l_low, indices=None):
        _, c, _, _ = l_encode.size()
        if indices is not None:
            l_encode = indices * F.interpolate(l_encode, size=l_low.size()[2:], mode='nearest')
        l_cat = torch.cat((l_encode, l_low), dim=1)
        return self.dconv(l_cat)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, BatchNorm2d_class):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
