"""
Author: Ikhyeon Cho
Link: https://github.com/Ikhyeon-Cho
File: LMSCNet.py
Date: 2024/11/2 18:50

Re-implementation of LMSCNet.
Reference: https://github.com/astra-vision/LMSCNet
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from models.nn_modules import Conv2DRelu


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


class LMSCNet(nn.Module):

    def __init__(self, num_classes, input_dims):
        '''
        LMSCNet architecture.
        '''
        super().__init__()

        self.num_classes = num_classes
        self.voxel_dims = input_dims  # [256, 256, 32]

        # Encoder modules
        CH_BASE = self.voxel_dims[2]  # 32
        CH_1_2 = int(CH_BASE*1.5)     # 48
        CH_1_4 = int(CH_BASE*2)       # 64
        CH_1_8 = int(CH_BASE*2.5)     # 80

        self.encoder_1_1 = nn.Sequential(  # CH_BASE: 32
            Conv2DRelu(CH_BASE, CH_BASE, kernel_size=3, padding=1, stride=1),
            Conv2DRelu(CH_BASE, CH_BASE, kernel_size=3, padding=1, stride=1))

        self.encoder_1_2 = nn.Sequential(  # CH_1_2: 48
            nn.MaxPool2d(2),
            Conv2DRelu(CH_BASE, CH_1_2, kernel_size=3, padding=1, stride=1),
            Conv2DRelu(CH_1_2, CH_1_2, kernel_size=3, padding=1, stride=1))

        self.encoder_1_4 = nn.Sequential(  # CH_1_4: 64
            nn.MaxPool2d(2),
            Conv2DRelu(CH_1_2, CH_1_4, kernel_size=3, padding=1, stride=1),
            Conv2DRelu(CH_1_4, CH_1_4, kernel_size=3, padding=1, stride=1))

        self.encoder_1_8 = nn.Sequential(  # CH_1_8: 80
            nn.MaxPool2d(2),
            Conv2DRelu(CH_1_4, CH_1_8, kernel_size=3, padding=1, stride=1),
            Conv2DRelu(CH_1_8, CH_1_8, kernel_size=3, padding=1, stride=1))

        # Output branch modules
        CH_OUT_1_8 = int(CH_BASE/8)  # 4
        CH_OUT_1_4 = int(CH_BASE/4)  # 8
        CH_OUT_1_2 = int(CH_BASE/2)  # 16

        self.conv_out_1_8 = nn.Conv2d(CH_1_8, CH_OUT_1_8,  # 80 -> 4
                                      kernel_size=3, padding=1, stride=1)
        self.conv_out_1_4 = nn.Conv2d(CH_1_4, CH_OUT_1_4,  # 64 -> 8
                                      kernel_size=3, padding=1, stride=1)
        self.conv_out_1_2 = nn.Conv2d(CH_1_2, CH_OUT_1_2,  # 48 -> 16
                                      kernel_size=3, padding=1, stride=1)

        dilation_rates = [1, 2, 3]
        self.seg_head_1_8 = SegmentationHead3d(
            1, 8, num_classes, dilation_rates)
        self.seg_head_1_4 = SegmentationHead3d(
            1, 8, num_classes, dilation_rates)
        self.seg_head_1_2 = SegmentationHead3d(
            1, 8, num_classes, dilation_rates)
        self.seg_head_1_1 = SegmentationHead3d(
            1, 8, num_classes, dilation_rates)

        # Upsampling modules
        self.upconv_1_8 = nn.ConvTranspose2d(CH_OUT_1_8, CH_OUT_1_8,  # spatial: 32 -> 64
                                             kernel_size=6, padding=2, stride=2)

        self.upconv_1_4 = nn.ConvTranspose2d(CH_OUT_1_4, CH_OUT_1_4,  # spatial: 64 -> 128
                                             kernel_size=6, padding=2, stride=2)

        self.upconv_1_2 = nn.ConvTranspose2d(CH_OUT_1_2, CH_OUT_1_2,  # spatial: 128 -> 256
                                             kernel_size=6, padding=2, stride=2)

        self.upconv_1_8_to_1_2 = nn.ConvTranspose2d(CH_OUT_1_8, CH_OUT_1_8,  # spatial: 32 -> 128
                                                    kernel_size=4, padding=0, stride=4)

        self.upconv_1_8_to_1_1 = nn.ConvTranspose2d(CH_OUT_1_8, CH_OUT_1_8,  # spatial: 32 -> 256
                                                    kernel_size=8, padding=0, stride=8)

        self.upconv_1_4_to_1_1 = nn.ConvTranspose2d(CH_OUT_1_4, CH_OUT_1_4,  # spatial: 64 -> 256
                                                    kernel_size=4, padding=0, stride=4)

        # Decoder modules
        self.conv_1_4 = nn.Conv2d((CH_OUT_1_8 + CH_1_4), CH_1_4,  # spatial: (32+64) -> 64 # why reduced to encoder channels?
                                  kernel_size=3, padding=1, stride=1)

        self.conv_1_2 = nn.Conv2d((CH_OUT_1_8 + CH_OUT_1_4 + CH_1_2), CH_1_2,  # spatial: (32+8+48) -> 48
                                  kernel_size=3, padding=1, stride=1)

        self.conv_1_1 = nn.Conv2d((CH_OUT_1_8 + CH_OUT_1_4 + CH_OUT_1_2 + CH_BASE), CH_BASE,  # spatial: (4+8+16+32) -> 32
                                  kernel_size=3, padding=1, stride=1)

    def forward(self, input_tensor, phase='train'):

        features = input_tensor.to(torch.float32)
        # Check if the input tensor has only 3 dimensions (no batch dimension)
        if len(features.shape) == 3:
            features = features.unsqueeze(0)
        features = features.permute(0, 3, 1, 2)  # pytorch channel order

        # Encoder pathway
        skip_1_1 = self.encoder_1_1(features)  # [bs, 32, 256, 256]
        skip_1_2 = self.encoder_1_2(skip_1_1)  # [bs, 48, 128, 128]
        skip_1_4 = self.encoder_1_4(skip_1_2)  # [bs, 64,  64,  64]
        skip_1_8 = self.encoder_1_8(skip_1_4)  # [bs, 80,  32,  32]

        # Bottleneck pathway
        # Scale 1:8 outputs
        output_1_8 = self.conv_out_1_8(skip_1_8)        # [bs, 4, 32, 32]
        seg_output_1_8 = self.seg_head_1_8(output_1_8)  # [bs,20,4,32,32]

        # Decoder pathway
        # Scale 1:4 outputs
        up_1_4 = self.upconv_1_8(output_1_8)                # [bs, 4, 64, 64]
        skip_conn_1_4 = torch.cat((up_1_4,                  # Skip connection
                                   skip_1_4), 1)            # [bs, 4+64, 64, 64]
        decoded_1_4 = F.relu(self.conv_1_4(skip_conn_1_4))  # [bs, 64, 64, 64]
        output_1_4 = self.conv_out_1_4(decoded_1_4)         # [bs,  8, 64, 64]
        seg_output_1_4 = self.seg_head_1_4(output_1_4)      # [bs,20,8,64, 64]

        # Scale 1:2 outputs
        up_1_2 = self.upconv_1_4(output_1_4)                # [bs, 8, 128, 128]
        up_1_8_to_1_2 = self.upconv_1_8_to_1_2(output_1_8)  # [bs, 4, 128, 128]
        skip_conn_1_2 = torch.cat((up_1_2,                  # Skip connection
                                   skip_1_2,
                                   up_1_8_to_1_2), 1)       # [bs, 8+48+4, 128, 128]
        decoded_1_2 = F.relu(self.conv_1_2(skip_conn_1_2))  # [bs,48,128,128]
        output_1_2 = self.conv_out_1_2(decoded_1_2)        # [bs, 16, 128, 128]
        seg_output_1_2 = self.seg_head_1_2(output_1_2)     # [bs,20,16,128,128]

        # Scale 1:1 outputs
        up_1_1 = self.upconv_1_2(output_1_2)               # [bs, 16, 256, 256]
        up_1_4_to_1_1 = self.upconv_1_4_to_1_1(output_1_4)  # [bs, 8, 256, 256]
        up_1_8_to_1_1 = self.upconv_1_8_to_1_1(output_1_8)  # [bs, 4, 256, 256]
        skip_conn_1_1 = torch.cat((up_1_1,                  # Skip connection
                                   skip_1_1,
                                   up_1_4_to_1_1,
                                   up_1_8_to_1_1), 1)      # [bs, 16+32+8+4, 256, 256]
        output_1_1 = F.relu(self.conv_1_1(skip_conn_1_1))  # [bs, 32, 256, 256]
        seg_output_1_1 = self.seg_head_1_1(output_1_1)     # [bs,20,32,256,256]

        if phase == 'train':
            return {
                'pred_semantic_1_1': seg_output_1_1.permute(0, 1, 3, 4, 2),
                'pred_semantic_1_2': seg_output_1_2.permute(0, 1, 3, 4, 2),
                'pred_semantic_1_4': seg_output_1_4.permute(0, 1, 3, 4, 2),
                'pred_semantic_1_8': seg_output_1_8.permute(0, 1, 3, 4, 2)
            }
        else:
            return seg_output_1_1.permute(0, 1, 3, 4, 2)


class LMSCNetLoss:
    def __init__(self, config: dict):
        self.num_classes = 20
        self.CE_Loss = nn.CrossEntropyLoss()

    def CE_Loss_1_1(self, preds: dict, targets: torch.Tensor):
        return self.CE_Loss(preds['pred_semantic_1_1'], targets)

    def CE_Loss_1_2(self, preds: dict, targets: torch.Tensor):
        return self.CE_Loss(preds['pred_semantic_1_2'], targets)

    def CE_Loss_1_4(self, preds: dict, targets: torch.Tensor):
        return self.CE_Loss(preds['pred_semantic_1_4'], targets)

    def CE_Loss_1_8(self, preds: dict, targets: torch.Tensor):
        return self.CE_Loss(preds['pred_semantic_1_8'], targets)


# if __name__ == '__main__':

#     # Dataset
#     from semantic_kitti_pytorch.data.datasets import SemanticKITTI_Completion
#     dataset = SemanticKITTI_Completion(
#         "/data/semanticKITTI/dataset/", phase='train')
#     sample_dict = dataset[0]

#     voxel_occ = sample_dict['occupancy']

#     # Model
#     model = LMSCNet(num_classes=20, input_dims=[256, 256, 32])
#     pred_1_1 = model(voxel_occ)
#     print(voxel_occ.shape)
#     print(pred_1_1.shape)
