

import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.init as int

import numpy as np
import torch
from torch import nn
from torch.nn import init


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=1):
        super().__init__()
        self.maxpool = nn.MaxPool2d(1)
        self.avgpool = nn.AvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class CBAM(nn.Module):

    def __init__(self, channel=512, reduction=1, kernel_size=3):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + residual


class SE(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) #对应Squeeze操作
        y = self.fc(y).view(b, c, 1, 1) #对应Excitation操作
        return x * y.expand_as(x)
def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


# spatial-spectral domain attention learning(SDL)
class DCSMA(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1, mode="ch*sp"):
        super(DCSMA, self).__init__()

        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        self.mode = mode

        self.conv_q_right = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_v_right = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                      bias=False)
        self.conv_up = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.conv_q_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)  # g
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)  # theta
        self.softmax_left = nn.Softmax(dim=2)

        self.reset_parameters()
        self.se = SE(channel=self.inplanes, reduction=1)
        self.cbam = CBAM(channel=self.inplanes, reduction=1, kernel_size=3)
        # new
        # self.conv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1,
        #                        bias=True)
        # todo conv改成kernel_size=1或5试试

    def reset_parameters(self):
        kaiming_init(self.conv_q_right, mode='fan_in')
        kaiming_init(self.conv_v_right, mode='fan_in')
        kaiming_init(self.conv_q_left, mode='fan_in')
        kaiming_init(self.conv_v_left, mode='fan_in')

        self.conv_q_right.inited = True
        self.conv_v_right.inited = True
        self.conv_q_left.inited = True
        self.conv_v_left.inited = True

    # HR spatial attention
    def spatial_attention(self, x):
        input_x = self.conv_v_right(x)
        batch, channel, height, width = input_x.size()

        input_x = input_x.view(batch, channel, height * width)
        context_mask = self.conv_q_right(x)
        context_mask = context_mask.view(batch, 1, height * width)
        context_mask = self.softmax_right(context_mask)

        context = torch.matmul(input_x, context_mask.transpose(1, 2))
        context = context.unsqueeze(-1)
        context = self.conv_up(context)
        # context = self.se(context)
        mask_ch = self.sigmoid(context)

        return mask_ch

    # HR spectral attention
    def spectral_attention(self, x):
        g_x = self.conv_q_left(x)
        batch, channel, height, width = g_x.size()
        avg_x = self.avg_pool(g_x)
        batch, channel, avg_x_h, avg_x_w = avg_x.size()

        avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)
        theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width)
        context = torch.matmul(avg_x, theta_x)
        context = self.softmax_left(context)
        context = context.view(batch, 1, height, width)

        mask_sp = self.sigmoid(context)

        return mask_sp

    def forward(self, x):
        mask_ch = self.spatial_attention(x)
        mask_sp = self.spectral_attention(x)
        if self.mode == "ch":
            out = x * mask_ch
        elif self.mode == "sp":
            out = x * mask_sp
        elif self.mode == "ch+sp":
            out = x * mask_ch + x * mask_sp
        elif self.mode == "ch*sp":
            out = x * mask_ch * mask_sp
            out = self.cbam(out)
        else:
            raise ValueError("DDA mode is unsupported")

        return out


if __name__ == "__main__":
    # #########################测试数据 ################################


    x = torch.ones(512, 512, 32, 32)    # print(model)
    model = DCSMA(512, 512)
    out = model(x)
    print(out.shape)

    # ##################################################################