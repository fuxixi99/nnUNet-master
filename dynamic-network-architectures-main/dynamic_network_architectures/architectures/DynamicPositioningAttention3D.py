# import torch
# from torch import Tensor, nn
# import nibabel as nib
# import math
# from torch.nn import functional as F
# import matplotlib.pyplot as plt
# import numpy as np
# from typing import List, Tuple, Type
# class DynamicPositioningAttention3D(nn.Module):
#     def __init__(self, in_channels, num_groups=8, log_file="C:/Users/B507/Desktop/hufei/SAM-Med3D-main/segment_anything/modeling/attention_log.txt"):
#         """
#         Dynamic Positioning Attention for 3D input.

#         Args:
#             in_channels (int): Number of input channels.
#             num_groups (int): Number of groups for channel grouping.
#             log_file (str): Path to the log file for saving attention details.
#         """
#         super(DynamicPositioningAttention3D, self).__init__()
#         self.in_channels = in_channels
#         self.num_groups = num_groups
#         self.log_file = log_file

#         # Linear projections for Q, K, V
#         self.query_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
#         self.key_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
#         self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)

#         # Output projection
#         self.output_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)

#     def forward(self, x):
#         B, C, D, H, W = x.shape

#         # Generate Q, K, V
#         Q = self.query_conv(x)  # Query projection
#         K = self.key_conv(x)    # Key projection
#         V = self.value_conv(x)  # Value projection

#         # Preserve important components based on average value selection
#         avg_Q = Q.mean(dim=(2, 3, 4), keepdim=True)
#         significant_mask = Q >= avg_Q
#         A_S = significant_mask.float()  # Binary mask for significant regions

#         # Edge detection using Laplacian operator
#         laplacian_kernel = torch.zeros((C, 1, 3, 3, 3), device=x.device)
#         laplacian_kernel[:, 0, 1, 1, 1] = 4  # Center weight
#         laplacian_kernel[:, 0, 1, 0, 1] = -1  # Top
#         laplacian_kernel[:, 0, 1, 2, 1] = -1  # Bottom
#         laplacian_kernel[:, 0, 0, 1, 1] = -1  # Left
#         laplacian_kernel[:, 0, 2, 1, 1] = -1  # Right

#         edge_detected = F.conv3d(A_S, weight=laplacian_kernel, padding=1, groups=C)
#         contour_locations = edge_detected > 0  # Binary mask for contours

#         # Create dynamic windows based on contour locations
#         dynamic_window = contour_locations.float()

#         # Apply dynamic windows to Q, K, V
#         Q_dw = Q * dynamic_window
#         K_dw = K * dynamic_window
#         V_dw = V * dynamic_window

#         # Compute attention scores
#         attention_scores = torch.einsum('bcxyz,bcxyz->bxyz', Q_dw, K_dw)
#         attention_scores = attention_scores / (C ** 0.5)
#         attention_weights = F.softmax(attention_scores, dim=-1)

#         # Attention application
#         attention_output = attention_weights.unsqueeze(1) * V_dw

#         # Log attention details
#         self.log_attention(contour_locations, dynamic_window)

#         # Output projection
#         out = self.output_conv(attention_output)
#         return out


#     def log_attention(self, attention_scores, dynamic_windows):
#         """Log the selected region coordinates and visualize them."""
#         for i, (score, window) in enumerate(zip(attention_scores, dynamic_windows)):
#             selected_coords = torch.nonzero(window, as_tuple=False).tolist()

#             # Save selected coordinates to the log file
#             with open(self.log_file, "a") as f:
#                 f.write(f"Batch {i} Selected Coordinates:/n{selected_coords}/n/n")

#             # Visualize the dynamic window as a heatmap (max projection)
#             heatmap = window.float().mean(dim=0).detach().cpu().numpy()
#             plt.figure(figsize=(8, 6))
#             plt.imshow(heatmap.max(axis=0), cmap="hot", interpolation="nearest")
#             plt.colorbar()
#             plt.title(f"Batch {i} Dynamic Window Heatmap")
#             plt.savefig(f"C:/Users/B507/Desktop/hufei/SAM-Med3D-main/segment_anything/modeling/dynamic_window_heatmap_batch_{i}.png")
#             plt.close()

# if __name__ == "__main__":

#     input_tensor = torch.randn(2, 1, 128, 128, 128)
#     model=DynamicPositioningAttention3D(in_channels=1)
#     model.eval()

#     # 运行模型
#     output = model(input_tensor)
#     print("Input shape:", input_tensor.shape)
#     print("Output shape:", output.shape)

import torch
from torch import nn
import torch.nn.functional as F
import SimpleITK as sitk
import os


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
        os.makedirs(save_dir, exist_ok=True)

        # Linear projections for Q, K, V
        self.query_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)

        # Output projection
        self.output_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)

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

        # Apply Laplacian filter for edge detection
        laplacian_kernel = torch.tensor([[[[0, 0, 0],
                                           [0, -1, 0],
                                           [0, 1, 0]],
                                          [[0, 0, 0],
                                           [-1, 4, -1],
                                           [0, 0, 0]],
                                          [[0, 0, 0],
                                           [0, -1, 0],
                                           [0, 1, 0]]]]).repeat(C, 1, 1, 1, 1).to(x.device)
        # 确保权重类型与 A_S 类型一致
        laplacian_kernel = laplacian_kernel.to(A_S.dtype)
        edge_detected = F.conv3d(A_S, weight=laplacian_kernel, padding=1, groups=C)
        contour_locations = edge_detected > 0  # Binary mask for contours

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
        self.save_attention_weights(attention_weights, save_name)

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


if __name__ == "__main__":
    # Path to input .nii.gz file
    nii_file_path = "C:/Users/B507/Desktop/hufei/SAM-Med3D-main/data/test/imagesTs/11data.nii.gz"

    # Load the image as a tensor
    input_tensor = load_nii_image_as_tensor(nii_file_path)

    # Initialize the model
    model = DynamicPositioningAttention3D(in_channels=1, save_dir="./attention_output").to("cuda")
    model.eval()

    # Run the model
    output = model(input_tensor, save_name="example_attention")
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output.shape)
