import torch
from torch import nn
import torch.nn.functional as F


import SimpleITK as sitk
import os
import nibabel as nib
import numpy as np
class EdgeEnhancementAttention3D(nn.Module):
    def __init__(self, in_channels, num_groups=8, log_file="attention_log.txt"):
        """
        边缘增强注意机制（3D）。

        Args:
            in_channels (int): 输入通道数。
            num_groups (int): 通道分组数。
            log_file (str): 保存注意力细节的日志文件路径。
        """
        super(EdgeEnhancementAttention3D, self).__init__()
        self.in_channels = in_channels
        self.num_groups = num_groups
        self.log_file = log_file

        # 定义 Query、Key 和 Value 的线性投影
        self.query_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)

        # 输出投影
        self.output_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x (Tensor): 输入特征图，形状为 (B, C, D, H, W)。

        Returns:
            Tensor: 输出特征图，形状与输入一致。
        """
        B, C, D, H, W = x.shape

        # 生成 Query、Key、Value
        Q = self.query_conv(x)  # Query 投影
        K = self.key_conv(x)    # Key 投影
        V = self.value_conv(x)  # Value 投影

        # 使用 Laplacian 算子进行边缘检测
        laplacian_kernel = torch.zeros((C, 1, 3, 3, 3), device=x.device)
        laplacian_kernel[:, 0, 1, 1, 1] = 6  # 中心权重
        laplacian_kernel[:, 0, 1, 0, 1] = -1  # 上
        laplacian_kernel[:, 0, 1, 2, 1] = -1  # 下
        laplacian_kernel[:, 0, 0, 1, 1] = -1  # 左
        laplacian_kernel[:, 0, 2, 1, 1] = -1  # 右
        laplacian_kernel = laplacian_kernel.to(x.dtype)

        edge_detected = F.conv3d(x, weight=laplacian_kernel, padding=1, groups=C)
        edge_mask = edge_detected > 0  # 边缘区域掩码

        # 动态窗口生成，用于增强边缘区域
        dynamic_window = edge_mask.float()
        dynamic_window = dynamic_window / dynamic_window.max()  # 归一化

        # 将动态窗口应用于 Q、K、V
        Q_dw = Q * dynamic_window
        K_dw = K * dynamic_window
        V_dw = V * dynamic_window

        # 计算注意力分数
        attention_scores = torch.einsum('bcxyz,bcxyz->bxyz', Q_dw, K_dw)
        attention_scores = attention_scores / (C ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 应用注意力权重
        attention_output = attention_weights.unsqueeze(1) * V_dw

        # 输出投影
        out = self.output_conv(attention_output)

        # 保存注意力权重用于可视化
        self.save_attention_weights(dynamic_window, edge_detected)
        return out

    def save_attention_weights(self, dynamic_window, edge_detected):
        """
        保存动态窗口和边缘检测结果为 nii.gz 格式，便于可视化。

        Args:
            dynamic_window (Tensor): 动态窗口。
            edge_detected (Tensor): 边缘检测结果。
        """


        dynamic_window_np = dynamic_window[0, 0].detach().cpu().numpy()
        edge_detected_np = edge_detected[0, 0].detach().cpu().numpy()

        # 保存为 NIfTI 文件
        nib.save(nib.Nifti1Image(dynamic_window_np, np.eye(4)), "./attention_output/dynamic_window.nii.gz")
        nib.save(nib.Nifti1Image(edge_detected_np, np.eye(4)), "./attention_output/edge_detected.nii.gz")
        print("Saved attention weights as NIfTI files.")

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

# 测试代码
if __name__ == "__main__":
    # Path to input .nii.gz file
    nii_file_path = "C:/Users/B507/Desktop/hufei/SAM-Med3D-main/data/test/imagesTs/11data.nii.gz"

    # Load the image as a tensor
    input_tensor = load_nii_image_as_tensor(nii_file_path)

    # Initialize the model
    model = EdgeEnhancementAttention3D(in_channels=1).to("cuda")
    model.eval()

    # Run the model
    output = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output.shape)
