import torch
import torch.nn as nn
import torch.nn.functional as F
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


# Example usage
if __name__ == "__main__":
    model = WaveletChannelAttention(in_channels=384)
    input_tensor = torch.randn(2, 384, 8, 8, 8)  # (B, C, D, H, W)
    output = model(input_tensor)
    print("Output shape:", output.shape)
