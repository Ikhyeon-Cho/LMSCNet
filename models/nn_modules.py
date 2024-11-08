import torch
import torch.nn.functional as F
import torch.nn as nn

'''
VGG-like 2D Convolution with ReLU activation
'''


class Conv2DRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv2DRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class Conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv2DBatchNormRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=[1, 6, 12, 18]):
        super(ASPP, self).__init__()
        self.branches = nn.ModuleList()

        # 1x1 Convolution branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        # Atrous Convolution branches with different dilation rates
        for rate in rates:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3,
                          padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

        # Image Pooling branch (global average pooling)
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Final 1x1 Convolution to combine features
        self.final_conv = nn.Sequential(
            nn.Conv2d(len(self.branches) * out_channels + out_channels,
                      out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[2:]  # Spatial size of input tensor

        # Apply branches
        outputs = [branch(x) for branch in self.branches]

        # Image pooling branch, followed by upsampling to match input size
        pooled = self.image_pool(x)
        pooled = F.interpolate(
            pooled, size=size, mode='bilinear', align_corners=False)
        outputs.append(pooled)

        # Concatenate all branches
        x = torch.cat(outputs, dim=1)

        # Final convolution to fuse features
        return self.final_conv(x)
