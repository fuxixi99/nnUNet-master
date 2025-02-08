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

def create_3d_log_kernel(ksize=3, sigma=1.0, device='cpu', dtype=torch.float32):
    """
    构造 3D Laplacian of Gaussian (LoG) 核
    公式参考:
      LoG(x,y,z) = ((x^2+y^2+z^2 - 3*sigma^2) / sigma^4) * exp(-(x^2+y^2+z^2) / (2*sigma^2))
    """
    # 计算中心位置
    center = ksize // 2
    # 创建坐标网格
    grid_range = torch.arange(ksize, dtype=dtype, device=device) - center
    z, y, x = torch.meshgrid(grid_range, grid_range, grid_range, indexing='ij')
    
    # 计算 x^2+y^2+z^2
    squared_distance = x**2 + y**2 + z**2
    
    # 计算 LoG 核（省略了常数因子）
    log_kernel = ((squared_distance - 3 * sigma**2) / (sigma**4)) * torch.exp(-squared_distance / (2 * sigma**2))
    
    # 为了数值稳定，通常需要将核归一化使得和接近 0（LoG 的和理论上应该为0）
    log_kernel = log_kernel - log_kernel.mean()
    
    return log_kernel

class DynamicPositioningAttention3D(nn.Module):
    def __init__(self, in_channels, num_groups=8, save_dir="./attention_output"):
        """
        Dynamic Positioning Attention for 3D input.

        Args:
            in_channels (int): Number of input channels.
            num_groups (int): Number of groups for channel grouping.
            save_dir (str): Directory to save attention weights as .nii.gz files.
        """
        super(DynamicPositioningAttention3D, self).__init__()
        self.in_channels = in_channels
        self.num_groups = num_groups
        self.save_dir = save_dir
        # os.makedirs(save_dir, exist_ok=True)

        # Linear projections for Q, K, V
        self.query_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)

        # Output projection
        self.output_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)

        # 通道注意力模块
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels//8, 1),
            nn.ReLU(),
            nn.Conv3d(in_channels//8, in_channels, 1),
            nn.Sigmoid()
        )

    
    def forward(self, x, save_name="attention_weights"):
        """
        Forward pass of the module.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, H, W).
            save_name (str): Base name for saving attention weights.

        Returns:
            torch.Tensor: Output tensor of shape (B, C, D, H, W).
        """
        B, C, D, H, W = x.shape

        # Generate Q, K, V
        Q = self.query_conv(x)  # Query projection
        K = self.key_conv(x)    # Key projection
        V = self.value_conv(x)  # Value projection

        # Compute average Q for masking
        # avg_Q = Q.mean(dim=(2, 3, 4), keepdim=True)
        # significant_mask = Q >= avg_Q
        # A_S = significant_mask.float()

        # 通道注意力增强的显著区域
        channel_weights = self.channel_att(Q)  # [B,C,1,1,1]
        weighted_Q = Q * channel_weights
        avg_Q = weighted_Q.mean(dim=(2,3,4), keepdim=True)
        A_S = (weighted_Q >= avg_Q).float()  # 加入通道权重

        # 假设 A_S 为输入特征图，形状为 [B, C, D, H, W]
        B, C, D, H, W = A_S.shape

        # 设置核大小和标准差
        ksize = 3      # 也可以使用更大尺寸，比如5或7
        sigma = 1.0

        # 构造 3D LoG 核
        log_kernel_3d = create_3d_log_kernel(ksize=ksize, sigma=sigma, device=A_S.device, dtype=A_S.dtype)
        # 调整形状为 5D 权重：[out_channels, in_channels/groups, D, H, W]
        log_kernel_3d = log_kernel_3d.view(1, 1, ksize, ksize, ksize).repeat(C, 1, 1, 1, 1)

        # 使用 conv3d 进行 3D 卷积，padding 设为 ksize//2 以保持尺寸不变，groups=C 表示每个通道独立卷积
        edge_detected = F.conv3d(A_S, weight=log_kernel_3d, padding=ksize//2, groups=C)

        # 根据需要生成二值轮廓掩码（例如：大于 0 的位置视为边缘）
        contour_locations = edge_detected > 0


        # Create dynamic windows based on contour locations
        dynamic_window = contour_locations.float()

        # Apply dynamic windows to Q, K, V
        Q_dw = Q * dynamic_window
        K_dw = K * dynamic_window
        V_dw = V * dynamic_window

        # Compute attention scores
        attention_scores = torch.einsum('bcxyz,bcxyz->bxyz', Q_dw, K_dw)
        attention_scores = attention_scores / (C ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Attention application
        attention_output = attention_weights.unsqueeze(1) * V_dw

        # Save attention weights as .nii.gz
        # self.save_attention_weights(attention_weights, save_name)

        # Output projection
        out = self.output_conv(attention_output)
        return out

    def save_attention_weights(self, attention_weights, save_name):
        """
        Save attention weights as .nii.gz images.

        Args:
            attention_weights (torch.Tensor): Attention weights of shape (B, D, H, W).
            save_name (str): Base name for saving the weights.
        """
        attention_weights_np = attention_weights.detach().cpu().numpy()

        for i in range(attention_weights_np.shape[0]):
            sitk_image = sitk.GetImageFromArray(attention_weights_np[i])
            file_name = os.path.join(self.save_dir, f"{save_name}_batch_{i}.nii.gz")
            sitk.WriteImage(sitk_image, file_name)
            print(f"Saved attention weights for batch {i} to {file_name}")


def load_nii_image_as_tensor(file_path, device="cuda"):
    """
    Load a .nii.gz file and convert it to a PyTorch tensor.

    Args:
        file_path (str): Path to the .nii.gz file.
        device (str): Device to load the tensor on, default is "cuda".

    Returns:
        torch.Tensor: 3D image tensor with shape (1, C, D, H, W).
    """
    # Load the image using SimpleITK
    sitk_image = sitk.ReadImage(file_path)
    image_array = sitk.GetArrayFromImage(sitk_image)  # Shape: (D, H, W)

    # Normalize and convert to PyTorch tensor
    image_tensor = torch.tensor(image_array, dtype=torch.float32, device=device)
    image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    return image_tensor



class DynamicLogPlainConvUNet(nn.Module):
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
            [DynamicPositioningAttention3D(in_channels=features) for features in self.encoder.output_channels]
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
    data = torch.rand((1, 1, 192, 192, 192))

    model = DynamicLogPlainConvUNet(
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
