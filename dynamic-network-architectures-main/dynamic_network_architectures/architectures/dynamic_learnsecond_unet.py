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

        # Learnable edge detection module
        self.edge_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels//2, 1, kernel_size=3, padding=1)
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
        avg_Q = Q.mean(dim=(2, 3, 4), keepdim=True)
        significant_mask = Q >= avg_Q
        A_S = significant_mask.float()

        # # Apply Laplacian filter for edge detection
        # laplacian_kernel = torch.tensor([[[[0, 0, 0],
        #                                    [0, -1, 0],
        #                                    [0, 1, 0]],
        #                                   [[0, 0, 0],
        #                                    [-1, 4, -1],
        #                                    [0, 0, 0]],
        #                                   [[0, 0, 0],
        #                                    [0, -1, 0],
        #                                    [0, 1, 0]]]]).repeat(C, 1, 1, 1, 1).to(x.device)
        # # 确保权重类型与 A_S 类型一致
        # laplacian_kernel = laplacian_kernel.to(A_S.dtype)
        # edge_detected = F.conv3d(A_S, weight=laplacian_kernel, padding=1, groups=C)
        # contour_locations = edge_detected > 0  # Binary mask for contours

        # 3. Learnable edge detection
        edge_features = self.edge_conv(A_S)  # [B,1,D,H,W]
        # dynamic_window = (edge_features > 0).float()  # 保持二值化
        edge_weights = torch.sigmoid(edge_features)  # [0,1] range

        # # Create dynamic windows based on contour locations
        # dynamic_window = contour_locations.float()
        # 4. Create dynamic window with edge guidance
        dynamic_window = edge_weights.expand(-1, C, -1, -1, -1)  # [B,C,D,H,W]

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



class DynamicLearnSecondPlainConvUNet(nn.Module):
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
        i=0
        # 增强 skip 连接
        enhanced_skips = []
        for skip, dyn in zip(skips, self.dynamic_net):
            skip = skip.float()  # 确保输入为 float32
            if i<2:
                ll = dyn(skip)  # 应用 FEM 模块
                enhanced_skip = ll + skip
            else:
                enhanced_skip = skip                
            enhanced_skips.append(enhanced_skip)
            i+=1

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

    model = DynamicLearnSecondPlainConvUNet(
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
