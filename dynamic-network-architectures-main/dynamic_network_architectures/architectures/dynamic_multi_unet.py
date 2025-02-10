from typing import Union, Type, List, Tuple

import torch
import torch
from torch import nn
import torch.nn.functional as F
import SimpleITK as sitk
import os
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.plain_conv_encoder import (
    PlainConvEncoder,
)
from dynamic_network_architectures.building_blocks.residual import (
    BasicBlockD,
    BottleneckD,
)
from dynamic_network_architectures.building_blocks.residual_encoders import (
    ResidualEncoder,
)
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.building_blocks.unet_residual_decoder import (
    UNetResDecoder,
)
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from dynamic_network_architectures.initialization.weight_init import (
    init_last_bn_before_add_to_0,
)
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from torch.cuda.amp import autocast


import torch
import torch.nn as nn
import torch.nn.functional as F

#########################################
# 1. 多尺度动态窗口模块（改进版：池化后统一插值）
#########################################
class MultiScaleDynamicWindow3D(nn.Module):
    """
    利用不同尺度的平均池化生成动态窗口，并融合为一个软权重掩码。
    通过插值确保所有尺度池化后的输出尺寸一致。
    """
    def __init__(self, in_channels, scales=[3, 5, 7]):
        """
        Args:
            in_channels (int): 输入特征的通道数。
            scales (list): 期望使用的平均池化核尺寸列表。
        """
        super(MultiScaleDynamicWindow3D, self).__init__()
        self.scales = scales
        # 拼接后通过1x1x1卷积融合各尺度信息，将通道数压缩为1，得到单通道权重图
        self.fuse_conv = nn.Conv3d(len(scales) * in_channels, 1, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入特征，形状为 (B, C, D, H, W)
        Returns:
            dynamic_window (torch.Tensor): 融合后的动态窗口权重，形状为 (B, 1, D, H, W)，归一化到 [0, 1]
        """
        pooled_features = []
        B, C, D, H, W = x.shape
        # 目标尺寸统一为输入的空间尺寸
        target_size = (D, H, W)
        for s in self.scales:
            # 动态确定每个维度上的池化核大小：
            kernel_d = s if s <= D else D
            kernel_h = s if s <= H else H
            kernel_w = s if s <= W else W
            # 计算每个维度上的 padding（采用 kernel//2）
            pad_d = kernel_d // 2
            pad_h = kernel_h // 2
            pad_w = kernel_w // 2

            # 执行平均池化
            pooled = F.avg_pool3d(
                x,
                kernel_size=(kernel_d, kernel_h, kernel_w),
                stride=1,
                padding=(pad_d, pad_h, pad_w)
            )
            # 无条件插值到目标尺寸，确保各尺度输出一致
            pooled = F.interpolate(pooled, size=target_size, mode='trilinear', align_corners=False)
            pooled_features.append(pooled)

        # 在通道维度上拼接各尺度特征
        concat_features = torch.cat(pooled_features, dim=1)
        # 融合多尺度信息
        dynamic_window = self.fuse_conv(concat_features)
        # Sigmoid归一化到 [0,1]
        dynamic_window = torch.sigmoid(dynamic_window)
        return dynamic_window

#########################################
# 2. 多尺度动态定位注意力模块（使用改进后的动态窗口模块）
#########################################
class MultiScaleDynamicPositioningAttention3D(nn.Module):
    def __init__(self, in_channels, num_groups=8, scales=[3, 5, 7]):
        """
        多尺度动态定位注意力模块，通过多尺度动态窗口对 Q、K、V 加权后计算注意力。
        
        Args:
            in_channels (int): 输入特征的通道数。
            num_groups (int): 分组数（预留参数，目前未使用）。
            scales (list): 多尺度动态窗口中使用的平均池化核尺寸列表。
        """
        super(MultiScaleDynamicPositioningAttention3D, self).__init__()
        self.in_channels = in_channels
        self.num_groups = num_groups
        
        # 生成 Query、Key、Value 的 1x1x1 卷积投影
        self.query_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.key_conv   = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        
        # 输出投影
        self.output_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        
        # 多尺度动态窗口模块，用于生成“软”权重掩码
        self.dynamic_window_module = MultiScaleDynamicWindow3D(in_channels, scales=scales)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入张量，形状 (B, C, D, H, W)
        Returns:
            out (torch.Tensor): 输出张量，形状 (B, C, D, H, W)
        """
        B, C, D, H, W = x.shape
        
        # 生成 Q, K, V 投影
        Q = self.query_conv(x)  # (B, C, D, H, W)
        K = self.key_conv(x)
        V = self.value_conv(x)
        
        # 利用 Q 特征生成多尺度动态窗口（也可以试验用 x 或其他中间特征）
        dynamic_window = self.dynamic_window_module(Q)  # (B, 1, D, H, W)
        
        # 利用动态窗口对 Q, K, V 进行加权，突出重要区域的特征
        Q_dw = Q * dynamic_window
        K_dw = K * dynamic_window
        V_dw = V * dynamic_window
        
        # 计算注意力分数：采用简化的内积方式，对每个位置计算 Q_dw 与 K_dw 的内积
        attention_scores = torch.einsum('bcxyz,bcxyz->bxyz', Q_dw, K_dw)
        attention_scores = attention_scores / (C ** 0.5)
        # 将注意力分数展平后进行 softmax，再恢复为原始空间尺寸
        attention_weights = F.softmax(attention_scores.view(B, -1), dim=-1).view(B, D, H, W)
        
        # 利用注意力权重调制 V_dw（扩展到通道维度）
        attention_output = attention_weights.unsqueeze(1) * V_dw
        
        # 输出投影
        out = self.output_conv(attention_output)
        return out



class DynamicMultiPlainConvUNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        nonlin_first: bool = False,
    ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, (
            "n_conv_per_stage must have as many entries as we have "
            f"resolution stages. here: {n_stages}. "
            f"n_conv_per_stage: {n_conv_per_stage}"
        )
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), (
            "n_conv_per_stage_decoder must have one less entries "
            f"as we have resolution stages. here: {n_stages} "
            f"stages, so it should have {n_stages - 1} entries. "
            f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        )
        self.encoder = PlainConvEncoder(
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_conv_per_stage,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
            return_skips=True,
            nonlin_first=nonlin_first,
        )
        self.decoder = UNetDecoder(
            self.encoder,
            num_classes,
            n_conv_per_stage_decoder,
            deep_supervision,
            nonlin_first=nonlin_first,
        )

        # 初始化 FEM 模块
        self.dynamic_net = nn.ModuleList(
            [MultiScaleDynamicPositioningAttention3D(in_channels=features) for features in self.encoder.output_channels]
        )

    def forward(self, x):
        # 编码阶段
        skips = self.encoder(x)

        # 增强 skip 连接
        enhanced_skips = []
        for skip, dyn in zip(skips, self.dynamic_net):
            skip = skip.float()  # 确保输入为 float32
            ll = dyn(skip)  # 应用 FEM 模块
            enhanced_skip = ll + skip
            enhanced_skips.append(enhanced_skip)

        # 解码阶段
        outputs = self.decoder(enhanced_skips)
        return outputs

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), (
            "just give the image size without color/feature channels or "
            "batch channel. Do not give input_size=(b, c, x, y(, z)). "
            "Give input_size=(x, y(, z))!"
        )
        return self.encoder.compute_conv_feature_map_size(
            input_size
        ) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)

if __name__ == "__main__":

    data = torch.rand((1, 1, 192, 192, 192))

    model = DynamicMultiPlainConvUNet(
        1,
        6,
        (32, 64, 125, 256, 320, 320),
        nn.Conv3d,
        3,
        (1, 2, 2, 2, 2, 2),
        (2, 2, 2, 2, 2, 2),
        2,
        (2, 2, 2, 2, 2),
        False,
        nn.BatchNorm3d,
        None,
        None,
        None,
        nn.ReLU,
        deep_supervision=True,
    )

    print(model.compute_conv_feature_map_size(data.shape[2:]))

    output = model(data)
    for i in output:
        print(i.shape)