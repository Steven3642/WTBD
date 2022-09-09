

import torch as t





class ICONN(t.nn.Module):
    def __init__(self, is_1x1conv=True):
        super(ICONN, self).__init__()
        self.is_1x1conv = is_1x1conv
        self.relu = t.nn.ReLU(inplace=True)
        self.conv0 = t.nn.Sequential(
            t.nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            t.nn.BatchNorm2d(8),
            t.nn.GELU(),
        )
        self.identical1 = t.nn.Sequential(
            t.nn.Conv2d(8, 32, kernel_size=1, stride=1, bias=False),
            t.nn.BatchNorm2d(32),
        )
        self.identical2 = t.nn.Sequential(
            t.nn.Conv2d(32, 128, kernel_size=1, stride=1, bias=False),
            t.nn.BatchNorm2d(128),
        )
        self.identical3 = t.nn.Sequential(
            t.nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False),
            t.nn.BatchNorm2d(256),
        )
        self.EACB1 = t.nn.Sequential(
            t.nn.Conv2d(8, 8, kernel_size=1),
            t.nn.GELU(),
            ACB_Conv2d(8, 32, groups=1, kernel_size=3, stride=1, padding=1),
            t.nn.BatchNorm2d(32),
            t.nn.GELU()
        )

        self.MaxPool = t.nn.Sequential(
            t.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.AvgPool = t.nn.Sequential(
            t.nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.EACB2 = t.nn.Sequential(
            t.nn.Conv2d(32, 32, kernel_size=1),
            t.nn.GELU(),
            ACB_Conv2d(32, 128, groups=1, kernel_size=3, stride=1, padding=1),
            t.nn.BatchNorm2d(128),
            t.nn.GELU()
        )
        self.EACB3 = t.nn.Sequential(
            t.nn.Conv2d(128, 128, kernel_size=1),
            t.nn.GELU(),
            ACB_Conv2d(128, 256, groups=1, kernel_size=3, stride=1, padding=1),
            t.nn.BatchNorm2d(256),
            t.nn.GELU()
        )

        self.conv1size_1 = t.nn.Sequential(
            t.nn.Conv2d(64, 32, kernel_size=1),
            t.nn.GELU(),
        )
        self.conv1size_2 = t.nn.Sequential(
            t.nn.Conv2d(256, 128, kernel_size=1),
            t.nn.GELU(),
        )
        self.conv1size_3 = t.nn.Sequential(
            t.nn.Conv2d(512, 256, kernel_size=1),
            t.nn.GELU(),
        )
        self.se = t.nn.Sequential(
            t.nn.AdaptiveAvgPool2d(1),
            t.nn.Conv2d(64, 64 // 16, kernel_size=1),
            t.nn.GELU(),
            t.nn.Conv2d(64 // 16, 64, kernel_size=1),
            t.nn.GELU()
        )
        self.se2 = t.nn.Sequential(
            t.nn.AdaptiveAvgPool2d(1),
            t.nn.Conv2d(256, 256 // 16, kernel_size=1),
            t.nn.GELU(),
            t.nn.Conv2d(256 // 16, 256, kernel_size=1),
            t.nn.GELU()
        )
        self.se3 = t.nn.Sequential(
            t.nn.AdaptiveAvgPool2d(1),
            t.nn.Conv2d(512, 512 // 16, kernel_size=1),
            t.nn.GELU(),
            t.nn.Conv2d(512 // 16, 512, kernel_size=1),
            t.nn.GELU()
        )

        self.fc_fayers = t.nn.Sequential(
            t.nn.Linear(7 * 7 * 256, 3),
            t.nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x1 = self.conv0(x)
        identical1 = self.identical1(x1)
        x1 = self.EACB1(x1)
        x1 = identical1 + x1
        mp1 = self.MaxPool(x1)
        ap1 = self.AvgPool(x1)
        x1 = t.cat([mp1, ap1], 1)
        x2 = self.se(x1)
        x1 = x1 * x2
        x1 = self.conv1size_1(x1)
        identical2 = self.identical2(x1)
        x1 = self.EACB2(x1)
        x1 = identical2 + x1
        mp2 = self.MaxPool(x1)
        ap2 = self.AvgPool(x1)
        x1 = t.cat([mp2, ap2], 1)
        x3 = self.se2(x1)
        x1 = x1 * x3
        x1 = self.conv1size_2(x1)
        identical3 = self.identical3(x1)
        x1 = self.EACB3(x1)
        x1 = identical3 + x1
        mp3 = self.MaxPool(x1)
        ap3 = self.AvgPool(x1)
        x1 = t.cat([mp3, ap3], 1)
        x4 = self.se3(x1)
        x1 = x1 * x4
        x1 = self.conv1size_3(x1)
        x1 = x1.view(-1, 7 * 7 * 256)
        x1 = self.fc_fayers(x1)
        return x1

class ACB_Conv2d(t.nn.Module):
    def __init__(self, in_channels, out_channels, groups, kernel_size=3, stride=1, padding=1, bias=False, ):
        super(ACB_Conv2d, self).__init__()
        self.groups = groups
        self.bias = bias  # 8        16
        self.conv = t.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=padding, bias=bias, groups=groups)
        self.ac1 = t.nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size),
                                   stride=stride, padding=(0, padding), bias=bias, groups=groups)
        self.ac2 = t.nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1),
                                   stride=stride, padding=(padding, 0), bias=bias, groups=groups)
        self.conv0 = t.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.fusedconv = t.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, bias=bias, groups=groups)

    def forward(self, x):
        if self.training:
            ac1 = self.ac1(x)
            ac2 = self.ac2(x)
            x = self.conv(x)
            return x + ac2 + ac1
        else:
            x = self.fusedconv(x)
            return x

    def train(self, mode=True):
        super().train(mode=mode)
        if mode is False:

            weight = self.conv.weight.cpu().detach().numpy()
            weight[:, :, 1:2, :] = weight[:, :, 1:2, :] + self.ac1.weight.cpu().detach().numpy()
            weight[:, :, :, 1:2] = weight[:, :, :, 1:2] + self.ac2.weight.cpu().detach().numpy()
            self.fusedconv.weight = t.nn.Parameter(t.FloatTensor(weight))

            if self.bias:
                bias = self.x_rest.bias.cpu().detach().numpy() + self.ac1_rest.cpu().detach().numpy() + self.conv.ac2_rest.cpu().detach().numpy()
                self.fusedconv.bias = t.nn.Parameter(t.FloatTensor(bias))
        if t.cuda.is_available():
            self.fusedconv = self.fusedconv.cuda()