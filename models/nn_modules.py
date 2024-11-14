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


class SegmentationHead3d(nn.Module):

    def __init__(self, planes_in, planes_hidden, num_classes, dilation_rates):
        '''
        3D Segmentation heads to retrieve semantic segmentation at each scale.
        Formed by Dim expansion, Conv3D, ASPP block, Conv3D.
        '''
        super().__init__()

        self.dilation_rates = dilation_rates

        # Initial 3D convolution
        self.initial_conv = nn.Conv3d(planes_in, planes_hidden,
                                      kernel_size=3, padding=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

        # ASPP Block
        self.aspp_conv1 = nn.ModuleList()
        self.aspp_bn1 = nn.ModuleList()
        self.aspp_conv2 = nn.ModuleList()
        self.aspp_bn2 = nn.ModuleList()

        for dil in dilation_rates:
            self.aspp_conv1.append(nn.Conv3d(planes_hidden, planes_hidden,
                                             kernel_size=3, padding=dil, dilation=dil, bias=False))
            self.aspp_bn1.append(nn.BatchNorm3d(planes_hidden))
            self.aspp_conv2.append(nn.Conv3d(planes_hidden, planes_hidden,
                                             kernel_size=3, padding=dil, dilation=dil, bias=False))
            self.aspp_bn2.append(nn.BatchNorm3d(planes_hidden))

        # Final classification layer
        self.voxel_classifier = nn.Conv3d(planes_hidden, num_classes,
                                          kernel_size=3, padding=1, stride=1)  # why kernel size is 3? not conv 1x1?

    def forward(self, completion_3d):

        # Dimension expansion for segmentation
        completion_3d = completion_3d[:, None, :, :, :]

        # Initial 3D Convolution
        completion_3d = self.relu(self.initial_conv(completion_3d))

        # First ASPP branch
        aspp_out = self.aspp_bn2[0](self.aspp_conv2[0](
            self.relu(self.aspp_bn1[0](self.aspp_conv1[0](completion_3d)))))

        # Remaining ASPP branches
        for i in range(1, len(self.dilation_rates)):
            branch_out = self.aspp_bn2[i](self.aspp_conv2[i]
                                          (self.relu(self.aspp_bn1[i](self.aspp_conv1[i](completion_3d)))))
            aspp_out += branch_out

        # Skip connection (residual) and final 3D Convolution
        completion_3d = self.relu(aspp_out + completion_3d)
        seg_completion_3d = self.voxel_classifier(completion_3d)

        return seg_completion_3d
