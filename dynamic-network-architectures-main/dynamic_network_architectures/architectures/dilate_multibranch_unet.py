from typing import Union, Type, List, Tuple

import torch
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

class DilationBranch3D(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 2, 4]):
        super().__init__()
        # 3D 空洞卷积分支
        self.dilated_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                          padding=d, dilation=d, bias=False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            ) for d in dilation_rates
        ])
        # 通道融合卷积
        self.fusion_conv = nn.Conv3d(len(dilation_rates) * out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        features = [conv(x) for conv in self.dilated_convs]
        fused = self.fusion_conv(torch.cat(features, dim=1))
        return fused

class HybridConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 普通 + 空洞混合路径
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, 
                      padding=5, dilation=5),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = self.skip(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x + identity

class MultiBranchDilationModule3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 分支1：普通卷积路径
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 分支2：纯空洞卷积路径
        self.branch2 = DilationBranch3D(in_channels, out_channels // 2)
        
        # 分支3：混合卷积路径
        self.branch3 = HybridConvBlock3D(in_channels, out_channels)
        
        # 特征融合
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(out_channels * 2 + out_channels // 2, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        fused = self.fusion_conv(torch.cat([b1, b2, b3], dim=1))
        return fused

class DilateMultiBranchPlainConvUNet(nn.Module):
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
        self.FEM = nn.ModuleList(
            [MultiBranchDilationModule3D(features, features) for features in self.encoder.output_channels]
        )

    def forward(self, x):
        # 编码阶段
        skips = self.encoder(x)

        # 增强 skip 连接
        enhanced_skips = []
        for skip, fem in zip(skips, self.FEM):
            skip = skip.float()  # 确保输入为 float32
            ll = fem(skip)  # 应用 FEM 模块
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

    model = DilateMultiBranchPlainConvUNet(
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

    # if False:
    #     import hiddenlayer as hl

    #     g = hl.build_graph(model, data,
    #                        transforms=None)
    #     g.save("network_architecture.pdf")
    #     del g

    # print(model.compute_conv_feature_map_size(data.shape[2:]))

    output = model(data)
    for i in output:
        print(i.shape)