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
from models.nn_modules import Conv2DRelu, SegmentationHead3d


class LMSCNet(nn.Module):

    def __init__(self, num_classes, input_dims):
        '''
        LMSCNet architecture.
        '''
        super().__init__()

        self.num_classes = num_classes
        self.voxel_dims = input_dims  # [256, 256, 32]

        #########################################
        # Encoder modules
        #########################################
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

        #########################################
        # Output branch modules
        #########################################
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

        #########################################
        # Upsampling modules
        #########################################
        # spatial: 32 -> 64
        self.upconv_1_8 = nn.ConvTranspose2d(CH_OUT_1_8, CH_OUT_1_8,
                                             kernel_size=6, padding=2, stride=2)
        # spatial: 64 -> 128
        self.upconv_1_4 = nn.ConvTranspose2d(CH_OUT_1_4, CH_OUT_1_4,
                                             kernel_size=6, padding=2, stride=2)
        # spatial: 128 -> 256
        self.upconv_1_2 = nn.ConvTranspose2d(CH_OUT_1_2, CH_OUT_1_2,
                                             kernel_size=6, padding=2, stride=2)
        # spatial: 32 -> 128
        self.upconv_1_8_to_1_2 = nn.ConvTranspose2d(CH_OUT_1_8, CH_OUT_1_8,
                                                    kernel_size=4, padding=0, stride=4)
        # spatial: 32 -> 256
        self.upconv_1_8_to_1_1 = nn.ConvTranspose2d(CH_OUT_1_8, CH_OUT_1_8,
                                                    kernel_size=8, padding=0, stride=8)
        # spatial: 64 -> 256
        self.upconv_1_4_to_1_1 = nn.ConvTranspose2d(CH_OUT_1_4, CH_OUT_1_4,
                                                    kernel_size=4, padding=0, stride=4)

        #########################################
        # Decoder modules
        #########################################
        # spatial: (32+64) -> 64  # TODO: Why reduced to encoder channels?
        self.conv_1_4 = nn.Conv2d((CH_OUT_1_8 + CH_1_4), CH_1_4,
                                  kernel_size=3, padding=1, stride=1)
        # spatial: (32+8+48) -> 48
        self.conv_1_2 = nn.Conv2d((CH_OUT_1_8 + CH_OUT_1_4 + CH_1_2), CH_1_2,
                                  kernel_size=3, padding=1, stride=1)
        # spatial: (4+8+16+32) -> 32
        self.conv_1_1 = nn.Conv2d((CH_OUT_1_8 + CH_OUT_1_4 + CH_OUT_1_2 + CH_BASE), CH_BASE,
                                  kernel_size=3, padding=1, stride=1)

        self.apply(self.weights_initializer)

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
                'pred': seg_output_1_1.permute(0, 1, 3, 4, 2),
                'pred_1_2': seg_output_1_2.permute(0, 1, 3, 4, 2),
                'pred_1_4': seg_output_1_4.permute(0, 1, 3, 4, 2),
                'pred_1_8': seg_output_1_8.permute(0, 1, 3, 4, 2)
            }
        else:
            return seg_output_1_1.permute(0, 1, 3, 4, 2)

    def weights_initializer(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            nn.init.kaiming_uniform_(
                m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, (Conv2DRelu, SegmentationHead3d)):
            # For custom modules, initialize their children
            for layer in m.children():
                self.weights_initializer(layer)


class LMSCNetMetrics:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.evaluator = {}

    def add_preds_1_1(self, preds: dict, targets: torch.Tensor):
        return self._update_confMat(preds['pred'], targets)

    def add_preds_1_2(self, preds: dict, targets: torch.Tensor):
        return self._update_confMat(preds['pred_1_2'], targets)

    def add_preds_1_4(self, preds: dict, targets: torch.Tensor):
        return self._update_confMat(preds['pred_1_4'], targets)

    def add_preds_1_8(self, preds: dict, targets: torch.Tensor):
        return self._update_confMat(preds['pred_1_8'], targets)

    def _update_confMat(self, preds: torch.Tensor, targets: torch.Tensor):
        pass


if __name__ == '__main__':

    # # Dataset
    # from semantic_kitti_pytorch.data.datasets import SemanticKITTI_Completion
    # dataset = SemanticKITTI_Completion(
    #     "/data/semanticKITTI/dataset/", phase='train')
    # sample_dict = dataset[0]

    # voxel_occ = sample_dict['occupancy']

    # Model
    model = LMSCNet(num_classes=20, input_dims=[256, 256, 32])
    print(model.num_classes)
    print(model.voxel_dims)
    # pred_1_1 = model(voxel_occ)
    # print(voxel_occ.shape)
    # print(pred_1_1.shape)
