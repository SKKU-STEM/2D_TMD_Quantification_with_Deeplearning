import torch
import torch.nn as nn
import torch.nn.functional as F

from model.unet_parts import *

class DenoisingNet(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1, hidden_channels = 64):
        super(DenoisingNet, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(in_channels = in_channels,
                                            out_channels = hidden_channels,
                                            kernel_size = 3,
                                            stride = 1,
                                            padding = 1,
                                            dilation = 1,
                                            bias = True),
                                  nn.BatchNorm2d(hidden_channels),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels = hidden_channels,
                                            out_channels = hidden_channels,
                                            kernel_size = 3,
                                            stride = 1,
                                            padding = 2,
                                            dilation = 2,
                                            bias = True),
                                  nn.BatchNorm2d(hidden_channels),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels = hidden_channels,
                                            out_channels = hidden_channels,
                                            kernel_size = 3,
                                            stride = 1,
                                            padding = 3,
                                            dilation = 3,
                                            bias = True),
                                  nn.BatchNorm2d(hidden_channels),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels = hidden_channels,
                                            out_channels = hidden_channels,
                                            kernel_size = 3,
                                            stride = 1,
                                            padding = 4,
                                            dilation = 4,
                                            bias = True),
                                  nn.BatchNorm2d(hidden_channels),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels = hidden_channels,
                                            out_channels = hidden_channels,
                                            kernel_size = 3,
                                            stride = 1,
                                            padding = 3,
                                            dilation = 3,
                                            bias = True),
                                  nn.BatchNorm2d(hidden_channels),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels = hidden_channels,
                                            out_channels = hidden_channels,
                                            kernel_size = 3,
                                            stride = 1,
                                            padding = 2,
                                            dilation = 2,
                                            bias = True),
                                  nn.BatchNorm2d(hidden_channels),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels = hidden_channels,
                                            out_channels = out_channels,
                                            kernel_size = 3,
                                            stride = 1,
                                            padding = 1,
                                            dilation = 1,
                                            bias = True),
                                  )
    def forward(self, x):
        noise = self.model(x)
        return x - noise
    

class PeakNet(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1, hidden_channels = 64):
        super(DenoisingNet, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(in_channels = in_channels,
                                            out_channels = hidden_channels,
                                            kernel_size = 3,
                                            stride = 1,
                                            padding = 1,
                                            dilation = 1,
                                            bias = True),
                                  nn.BatchNorm2d(hidden_channels),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels = hidden_channels,
                                            out_channels = hidden_channels,
                                            kernel_size = 3,
                                            stride = 1,
                                            padding = 2,
                                            dilation = 2,
                                            bias = True),
                                  nn.BatchNorm2d(hidden_channels),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels = hidden_channels,
                                            out_channels = hidden_channels,
                                            kernel_size = 3,
                                            stride = 1,
                                            padding = 3,
                                            dilation = 3,
                                            bias = True),
                                  nn.BatchNorm2d(hidden_channels),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels = hidden_channels,
                                            out_channels = hidden_channels,
                                            kernel_size = 3,
                                            stride = 1,
                                            padding = 4,
                                            dilation = 4,
                                            bias = True),
                                  nn.BatchNorm2d(hidden_channels),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels = hidden_channels,
                                            out_channels = hidden_channels,
                                            kernel_size = 3,
                                            stride = 1,
                                            padding = 3,
                                            dilation = 3,
                                            bias = True),
                                  nn.BatchNorm2d(hidden_channels),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels = hidden_channels,
                                            out_channels = hidden_channels,
                                            kernel_size = 3,
                                            stride = 1,
                                            padding = 2,
                                            dilation = 2,
                                            bias = True),
                                  nn.BatchNorm2d(hidden_channels),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels = hidden_channels,
                                            out_channels = out_channels,
                                            kernel_size = 3,
                                            stride = 1,
                                            padding = 1,
                                            dilation = 1,
                                            bias = True),
                                  )
    def forward(self, x):
        noise = self.model(x)
        return x - noise
    

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
