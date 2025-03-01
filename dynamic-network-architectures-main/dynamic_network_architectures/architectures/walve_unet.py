from typing import Union, Type, List, Tuple

import torch
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.building_blocks.unet_residual_decoder import UNetResDecoder
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
import pywt
import torch.nn.functional as F
class WaveletTransform3D(nn.Module):
    """
    Module to perform 3D Haar Wavelet Transform.
    Retains only the LL (low-frequency) component.
    """
    def __init__(self, wavelet='haar'):
        super(WaveletTransform3D, self).__init__()
        self.wavelet = wavelet

    def forward(self, x):
        x_np = x.detach().cpu().numpy()
        coeffs = pywt.dwtn(x_np, self.wavelet, axes=(2, 3, 4))
        LL = coeffs['aaa']
        LL_tensor = torch.tensor(LL, dtype=x.dtype, device=x.device)
        if len(LL_tensor.shape) == 4:
            LL_tensor = LL_tensor.unsqueeze(1)
        # return LL_tensor.squeeze(1)
        return LL_tensor


class WalvePlainConvUNet(nn.Module):
    def __init__(self,
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
                 nonlin_first: bool = False
                 ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_conv_per_stage: {n_conv_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = PlainConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                        n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                        nonlin_first=nonlin_first)
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                   nonlin_first=nonlin_first)
        self.wavelet_transform = WaveletTransform3D()

        # Pooling
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=(2, 2, 2))
        self.conv3d1 = nn.Conv3d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3d2 = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3d3 = nn.Conv3d(in_channels=250, out_channels=125, kernel_size=3, stride=1, padding=1)
        self.conv3d4 = nn.Conv3d(in_channels=512, out_channels=250, kernel_size=3, stride=1, padding=1)
        self.conv3d5 = nn.Conv3d(in_channels=640, out_channels=320, kernel_size=3, stride=1, padding=1)
        self.conv3d6 = nn.Conv3d(in_channels=640, out_channels=320, kernel_size=3, stride=1, padding=1)
        self.conv3d=[]
        self.conv3d.append(self.conv3d1)
        self.conv3d.append(self.conv3d2)
        self.conv3d.append(self.conv3d3)
        self.conv3d.append(self.conv3d4)
        self.conv3d.append(self.conv3d5)
        self.conv3d.append(self.conv3d6)


    def forward(self, x):
        skips = self.encoder(x)
        enhanced_skips = []
        i=0
        for skip, conv3d in zip(skips,self.conv3d):
            i+=1
            ll = self.wavelet_transform(skip)
            print("ll.shape",ll.shape)
            print("skip.shape",skip.shape)
            print("self.pool1(ll).shape",self.pool1(skip).shape)
            
            if i<4:
                enhanced_skip = torch.cat([ll, self.pool1(skip)], dim=1)
                print("before enhanced_skip.shape",enhanced_skip.shape)
                enhanced_skip=conv3d(enhanced_skip)
                print("after enhanced_skip.shape",enhanced_skip.shape)
            else:
                enhanced_skip =self.pool1(skip)
                print("enhanced_skip.shape",enhanced_skip.shape)
            
            enhanced_skips.append(enhanced_skip)
        # enhanced_skips = []

        # # 进行多次小波变换
        # ll1 = self.wavelet_transform(x)  # 第一次小波变换
        # enhanced_skips.append(ll1)  # 保存第一次小波低频分量

        # # 第二次小波变换
        # ll2 = self.wavelet_transform(ll1)
        # enhanced_skips.append(ll2)

        # # 第三次小波变换（可以根据需要调整次数）
        # ll3 = self.wavelet_transform(ll2)
        # enhanced_skips.append(ll3)
        

        # # 将增强的特征通过编码器和解码器处理
        # skips = self.encoder(x)  # 编码原始输入
        # combined_skips = []
        
        # # 对应每层 skip 添加增强特征
        # for i in range(3):
        #     print(i,"self.pool1(skips[i].shape",self.pool1(skips[i]).shape)
        #     print(i,"enhanced_skip.shape",enhanced_skips[i].shape)
        #     # 特征融合：将小波特征与编码特征拼接
        #     combined_skip = enhanced_skips[i]+ self.pool1(skips[i])
        #     print(i,"combined_skip.shape",combined_skip.shape)
        #     combined_skips.append(combined_skip)
        
        # for i in range(3,6):
        #     combined_skips.append(skips[i])
        #     print(i,"combined_skip.shape",combined_skips[i].shape)

        # 解码阶段
        # 解码阶段，decoder 返回一个列表（深度监督输出）
        outputs = self.decoder(enhanced_skips)

        upsampled_outputs = []  # 用于存储每层上采样后的结果

        if isinstance(outputs, list):  # 如果 decoder 返回多个输出
            for output in outputs:
                # 对每层输出进行上采样
                upsampled_output = F.interpolate(output, scale_factor=2, mode='trilinear', align_corners=False)
                upsampled_outputs.append(upsampled_output)
        else:
            # 如果 decoder 返回单个张量，直接上采样
            upsampled_outputs.append(F.interpolate(outputs, scale_factor=2, mode='trilinear', align_corners=False))

        # 返回所有上采样的结果
        return upsampled_outputs

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)


class ResidualEncoderUNet(nn.Module):
    def __init__(self,
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
                 stem_channels: int = None
                 ):
        super().__init__()
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_blocks_per_stage: {n_blocks_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = ResidualEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                       n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                       dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
                                       return_skips=True, disable_default_stem=False, stem_channels=stem_channels)
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                                "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)


class ResidualUNet(nn.Module):
    def __init__(self,
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
                 stem_channels: int = None
                 ):
        super().__init__()
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_blocks_per_stage: {n_blocks_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = ResidualEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                       n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                       dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
                                       return_skips=True, disable_default_stem=False, stem_channels=stem_channels)
        self.decoder = UNetResDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                                "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)


if __name__ == '__main__':
    data = torch.rand((1, 1, 192, 192, 192))

    model = WalvePlainConvUNet(1, 6, (32, 64, 125, 256, 320, 320), nn.Conv3d, 3, (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2), 2,
                                (2, 2, 2, 2, 2), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=True)

    # # if False:
    # #     import hiddenlayer as hl

    # #     g = hl.build_graph(model, data,
    # #                        transforms=None)
    # #     g.save("network_architecture.pdf")
    # #     del g

    # print(model.compute_conv_feature_map_size(data.shape[2:]))

    output=model(data)
    for i in output:
        print(i.shape)
        print("\n")

