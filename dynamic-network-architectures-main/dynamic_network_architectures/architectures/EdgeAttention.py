import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableEdgeAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Query/Key/Value projections
        self.query_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        
        # Learnable edge detection module
        self.edge_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels//2, 1, kernel_size=3, padding=1)
        )
        
        # Output projection
        self.output_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        
        # Channel attention for significance masking
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels//8, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels//8, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x, save_name=None):
        B, C, D, H, W = x.shape

        # 1. Generate Q, K, V
        Q = self.query_conv(x)
        K = self.key_conv(x)
        V = self.value_conv(x)

        # 2. Enhanced significance masking with channel attention
        channel_weights = self.channel_att(Q)  # [B,C,1,1,1]
        Q_weighted = Q * channel_weights
        avg_Q = Q_weighted.mean(dim=(2,3,4), keepdim=True)
        significant_mask = (Q_weighted >= avg_Q).float()

        # 3. Learnable edge detection
        edge_features = self.edge_conv(significant_mask)  # [B,1,D,H,W]
        edge_weights = torch.sigmoid(edge_features)  # [0,1] range
        
        # 4. Create dynamic window with edge guidance
        dynamic_window = edge_weights.expand(-1, C, -1, -1, -1)  # [B,C,D,H,W]
        
        # 5. Apply dynamic window to features
        Q_edge = Q * dynamic_window
        K_edge = K * dynamic_window
        V_edge = V * dynamic_window

        # 6. Compute attention scores
        attention_scores = torch.einsum('bcxyz,bcxyz->bxyz', Q_edge, K_edge)
        attention_scores = attention_scores / (C ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 7. Apply attention to values
        attention_output = attention_weights.unsqueeze(1) * V_edge

        # 8. Final projection
        out = self.output_conv(attention_output)
        return out

# 使用示例
# x = torch.randn(2, 64, 32, 32, 32)  # (B,C,D,H,W)
# module = LearnableEdgeAttention(64)
# output = module(x)

class LearnableDynamicEdgeAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Query/Key/Value projections
        self.query_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        
        # Learnable edge detection module
        self.edge_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels//2, 1, kernel_size=3, padding=1)
        )
        
        # Output projection
        self.output_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        
        # Channel attention for significance masking
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels//8, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels//8, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x, save_name=None):
        B, C, D, H, W = x.shape

        # 1. Generate Q, K, V
        Q = self.query_conv(x)
        K = self.key_conv(x)
        V = self.value_conv(x)

        # 2. Enhanced significance masking with channel attention
        channel_weights = self.channel_att(Q)  # [B,C,1,1,1]
        Q_weighted = Q * channel_weights
        avg_Q = Q_weighted.mean(dim=(2,3,4), keepdim=True)
        significant_mask = (Q_weighted >= avg_Q).float()

        # 3. Learnable edge detection
        edge_features = self.edge_conv(significant_mask)  # [B,1,D,H,W]
        edge_weights = torch.sigmoid(edge_features)  # [0,1] range
        
        # 4. Create dynamic window with edge guidance
        dynamic_window = edge_weights.expand(-1, C, -1, -1, -1)  # [B,C,D,H,W]
        
        # 5. Apply dynamic window to features
        Q_edge = Q * dynamic_window
        K_edge = K * dynamic_window
        V_edge = V * dynamic_window

        # 6. Compute attention scores
        attention_scores = torch.einsum('bcxyz,bcxyz->bxyz', Q_edge, K_edge)
        attention_scores = attention_scores / (C ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 7. Apply attention to values
        attention_output = attention_weights.unsqueeze(1) * V_edge

        # 8. Final projection
        out = self.output_conv(attention_output)
        return out
