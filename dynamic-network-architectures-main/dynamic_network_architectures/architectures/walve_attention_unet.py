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


import pywt

class WaveletTransform3D:
    """
    A utility class to perform 3D discrete wavelet transform (DWT) and inverse DWT.
    """
    def __init__(self, wavelet='haar'):
        self.wavelet = pywt.Wavelet(wavelet)

    def dwt3d(self, x):
        """
        Perform 3D discrete wavelet transform.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
        
        Returns:
            Low-frequency and high-frequency components.
        """
        B, C, D, H, W = x.shape

        # Convert tensor to numpy after detaching from computation graph
        coeffs = [
            pywt.dwtn(x[b, c].detach().cpu().numpy(), self.wavelet, mode='symmetric')
            for b in range(B) for c in range(C)
        ]
        
        # Extract low-frequency and high-frequency components
        low_freq = torch.stack([
            torch.tensor(c['aaa'], device=x.device, dtype=torch.float32) for c in coeffs
        ])
        high_freq = torch.stack([
            torch.cat([
                torch.tensor(c[k], device=x.device, dtype=torch.float32).unsqueeze(0)
                for k in ('aad', 'ada', 'add', 'daa', 'dad', 'dda', 'ddd')
            ], dim=0) for c in coeffs
        ])
        low_freq = low_freq.view(B, C, *low_freq.shape[1:])
        high_freq = high_freq.view(B, C, -1, *low_freq.shape[2:])
        return low_freq, high_freq


class ChannelAttention(nn.Module):
    """
    Channel attention mechanism to enhance feature maps.
    """
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y

class FeatureFusion(nn.Module):
    """
    Feature fusion module to combine low- and high-frequency components.
    """
    def __init__(self, low_channels, high_channels):
        super(FeatureFusion, self).__init__()
        self.conv = nn.Conv3d(low_channels + high_channels, low_channels, kernel_size=1)
        self.bn = nn.BatchNorm3d(low_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, low_freq, high_freq):
        # Concatenate along channel dimension
        x = torch.cat([low_freq, high_freq], dim=1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class WaveletChannelAttention(nn.Module):
    """
    Model combining wavelet transform and channel attention for 3D CT images.
    """
    def __init__(self, in_channels, wavelet='haar', reduction=16):
        super(WaveletChannelAttention, self).__init__()
        self.wavelet_transform = WaveletTransform3D(wavelet=wavelet)
        self.low_freq_attention = ChannelAttention(in_channels, reduction)
        self.high_freq_attention = ChannelAttention(in_channels * 7, reduction)  # Adapted for high frequency

        # Pass the correct low and high frequency channels to FeatureFusion
        self.feature_fusion = FeatureFusion(low_channels=in_channels, high_channels=in_channels * 7)

    def forward(self, x):
        # Perform wavelet transform
        low_freq, high_freq = self.wavelet_transform.dwt3d(x)

        # Adjust high_freq shape to (B, C * 7, D, H, W)
        B, C, N, D, H, W = high_freq.shape
        high_freq = high_freq.view(B, C * N, D, H, W)

        # Apply channel attention to low- and high-frequency components
        low_freq = self.low_freq_attention(low_freq)
        high_freq = self.high_freq_attention(high_freq)

        # Fuse the enhanced low- and high-frequency components
        fused_features = self.feature_fusion(low_freq, high_freq)

        return fused_features



class WalveAttentionPlainConvUNet(nn.Module):
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
        self.walveattention_net = nn.ModuleList(
            [WaveletChannelAttention(in_channels=features) for features in self.encoder.output_channels]
        )
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=(2, 2, 2))

    def forward(self, x):
        # 编码阶段
        skips = self.encoder(x)

        # 增强 skip 连接
        enhanced_skips = []
        for skip, wlve in zip(skips, self.walveattention_net):
            skip = skip.float()  # 确保输入为 float32
            ll = wlve(skip)  # 应用 FEM 模块
            # 上采样 从（4，4，4）上采样到（8，8，8）
            ll = F.interpolate(ll, scale_factor=2, mode='trilinear', align_corners=False)
            enhanced_skip = ll + skip
            enhanced_skips.append(enhanced_skip)

        # 解码阶段
        outputs = self.decoder(enhanced_skips)
        return outputs
        # upsampled_outputs = []  # 用于存储每层上采样后的结果

        # if isinstance(outputs, list):  # 如果 decoder 返回多个输出
        #     for output in outputs:
        #         # 对每层输出进行上采样
        #         upsampled_output = F.interpolate(output, scale_factor=2, mode='trilinear', align_corners=False)
        #         upsampled_outputs.append(upsampled_output)
        # else:
        #     # 如果 decoder 返回单个张量，直接上采样
        #     upsampled_outputs.append(F.interpolate(outputs, scale_factor=2, mode='trilinear', align_corners=False))

        # # 返回所有上采样的结果
        # return upsampled_outputs

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


class ResidualEncoderUNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
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
        block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
        bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
        stem_channels: int = None,
    ):
        super().__init__()
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, (
            "n_blocks_per_stage must have as many entries as we have "
            f"resolution stages. here: {n_stages}. "
            f"n_blocks_per_stage: {n_blocks_per_stage}"
        )
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), (
            "n_conv_per_stage_decoder must have one less entries "
            f"as we have resolution stages. here: {n_stages} "
            f"stages, so it should have {n_stages - 1} entries. "
            f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        )
        self.encoder = ResidualEncoder(
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_blocks_per_stage,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
            block,
            bottleneck_channels,
            return_skips=True,
            disable_default_stem=False,
            stem_channels=stem_channels,
        )
        self.decoder = UNetDecoder(
            self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision
        )

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

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
        init_last_bn_before_add_to_0(module)


class ResidualUNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
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
        block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
        bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
        stem_channels: int = None,
    ):
        super().__init__()
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, (
            "n_blocks_per_stage must have as many entries as we have "
            f"resolution stages. here: {n_stages}. "
            f"n_blocks_per_stage: {n_blocks_per_stage}"
        )
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), (
            "n_conv_per_stage_decoder must have one less entries "
            f"as we have resolution stages. here: {n_stages} "
            f"stages, so it should have {n_stages - 1} entries. "
            f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        )
        self.encoder = ResidualEncoder(
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_blocks_per_stage,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
            block,
            bottleneck_channels,
            return_skips=True,
            disable_default_stem=False,
            stem_channels=stem_channels,
        )
        self.decoder = UNetResDecoder(
            self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision
        )

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

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
        init_last_bn_before_add_to_0(module)


if __name__ == "__main__":
    data = torch.rand((1, 1, 128, 128, 128))

    model = WalveAttentionPlainConvUNet(
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

    print(model.compute_conv_feature_map_size(data.shape[2:]))

    output = model(data)
    for i in output:
        print(i.shape)