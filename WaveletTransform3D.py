# import torch
# import torch.nn as nn
# import pywt
# import torch.nn.functional as F

# class WaveletTransform3D(nn.Module):
#     """
#     Module to perform 3D Haar Wavelet Transform.
#     Retains only the LL (low-frequency) component.
#     """
#     def __init__(self, wavelet='haar'):
#         super(WaveletTransform3D, self).__init__()
#         self.wavelet = wavelet

#     def forward(self, x):
#         # Convert the input tensor to numpy array for wavelet transform
#         x_np = x.detach().cpu().numpy()

#         # Apply 3D wavelet transform
#         coeffs = pywt.dwtn(x_np, self.wavelet, axes=(2, 3, 4))
#         LL = coeffs['aaa']  # Low-frequency component

#         # Convert back to tensor
#         LL_tensor = torch.tensor(LL, dtype=x.dtype, device=x.device)

#         # Resize LL to match input dimensions if necessary
#         # Check the input tensor shape
#         # print("Original LL_tensor shape:", LL_tensor.shape)

#         # Add the channel dimension if missing
#         if len(LL_tensor.shape) == 4:  # If input shape is (N, D, H, W)
#             LL_tensor = LL_tensor.unsqueeze(1)  # Add channel dimension: (N, 1, D, H, W)
#             # print("After unsqueeze, LL_tensor shape:", LL_tensor.shape)

#         # Interpolate to the target size
#         # LL_tensor = F.interpolate(LL_tensor, size=(128, 128, 128), mode="trilinear", align_corners=False)
#         # print("After interpolation, LL_tensor shape:", LL_tensor.shape)

#         # If you need to remove the channel dimension after interpolation
#         LL_tensor = LL_tensor.squeeze(1)  # Optional: Remove channel dimension
#         # print("After squeeze, LL_tensor shape:", LL_tensor.shape)


#         return LL_tensor.squeeze(1)


# class ConvBlock3D(nn.Module):
#     """
#     Basic Convolutional Block with BatchNorm and ReLU.
#     """
#     def __init__(self, in_channels, out_channels):
#         super(ConvBlock3D, self).__init__()
#         self.block = nn.Sequential(
#             nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm3d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm3d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.block(x)


# class WaveletUNet3D(nn.Module):
#     """
#     3D UNet with Wavelet Transform integrated into the encoder path.
#     """
#     def __init__(self, in_channels, out_channels):
#         super(WaveletUNet3D, self).__init__()
#         self.wavelet_transform = WaveletTransform3D()

#         # Encoder
#         self.enc1 = ConvBlock3D(in_channels, 32)
#         self.enc2 = ConvBlock3D(32, 64)
#         self.enc3 = ConvBlock3D(64, 128)
#         self.enc4 = ConvBlock3D(128, 256)

#         # Pooling
#         self.pool = nn.MaxPool3d(2)

#         # Bottleneck
#         self.bottleneck = ConvBlock3D(256, 512)

#         # Decoder
#         self.up4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
#         self.dec4 = ConvBlock3D(512, 256)
#         self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
#         self.dec3 = ConvBlock3D(256, 128)
#         self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
#         self.dec2 = ConvBlock3D(128, 64)
#         self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
#         self.dec1 = ConvBlock3D(64, 32)

#         # Final Convolution
#         self.final_conv = nn.Conv3d(32, out_channels, kernel_size=1)

#     def forward(self, x):
#         # Encoder path with wavelet transform
#         enc1 = self.enc1(x)
#         ll1 = self.wavelet_transform(enc1)


#         enc2 = self.enc2(self.pool(enc1) + ll1)
#         ll2 = self.wavelet_transform(enc2)
#         enc3 = self.enc3(self.pool(enc2) + ll2)
#         ll3 = self.wavelet_transform(enc3)
#         enc4 = self.enc4(self.pool(enc3) + ll3)

#         # Bottleneck
#         bottleneck = self.bottleneck(self.pool(enc4))

#         # Decoder path
#         up4 = self.up4(bottleneck)
#         dec4 = self.dec4(torch.cat((up4, enc4), dim=1))
#         up3 = self.up3(dec4)
#         dec3 = self.dec3(torch.cat((up3, enc3), dim=1))
#         up2 = self.up2(dec3)
#         dec2 = self.dec2(torch.cat((up2, enc2), dim=1))
#         up1 = self.up1(dec2)
#         dec1 = self.dec1(torch.cat((up1, enc1), dim=1))

#         # Final output
#         out = self.final_conv(dec1)
#         return out



import torch
import torch.nn as nn
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
        return LL_tensor.squeeze(1)

class ConvBlock3D(nn.Module):
    """
    Basic Convolutional Block with InstanceNorm and LeakyReLU.
    """
    def __init__(self, in_channels, out_channels, conv_bias=True, norm_op=nn.InstanceNorm3d, norm_kwargs=None, nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvBlock3D, self).__init__()
        norm_kwargs = norm_kwargs or {"eps": 1e-05, "affine": True}
        nonlin_kwargs = nonlin_kwargs or {"inplace": True}
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=conv_bias),
            norm_op(out_channels, **norm_kwargs),
            nonlin(**nonlin_kwargs),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=conv_bias),
            norm_op(out_channels, **norm_kwargs),
            nonlin(**nonlin_kwargs)
        )

    def forward(self, x):
        return self.block(x)

class WaveletUNet3D(nn.Module):
    """
    3D UNet with Wavelet Transform integrated into the encoder path.
    """
    def __init__(self, in_channels, out_channels):
        super(WaveletUNet3D, self).__init__()
        self.wavelet_transform = WaveletTransform3D()

        # Encoder
        self.enc1 = ConvBlock3D(in_channels, 32)
        self.enc2 = ConvBlock3D(32, 64)
        self.enc3 = ConvBlock3D(64, 128)
        self.enc4 = ConvBlock3D(128, 256)
        self.enc5 = ConvBlock3D(256, 320)
        self.enc6 = ConvBlock3D(320, 320)

        # Pooling
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=(2, 2, 2))
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=(2, 2, 2))
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=(2, 2, 2))
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=(2, 2, 2))
        self.pool5 = nn.MaxPool3d(kernel_size=2, stride=(2, 2, 2))

        # Bottleneck
        self.bottleneck = ConvBlock3D(320, 512)

        # Decoder
        self.up5 = nn.ConvTranspose3d(512, 320, kernel_size=2, stride=(1, 2, 2))
        self.dec5 = ConvBlock3D(640, 320)
        self.up4 = nn.ConvTranspose3d(320, 256, kernel_size=2, stride=2)
        self.dec4 = ConvBlock3D(512, 256)
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(256, 128)
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(128, 64)
        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(64, 32)

        # Final Convolution
        self.final_conv = nn.Conv3d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path with wavelet transform
        enc1 = self.enc1(x)
        ll1 = self.wavelet_transform(enc1)

        enc2 = self.enc2(self.pool1(enc1) + ll1)
        ll2 = self.wavelet_transform(enc2)

        enc3 = self.enc3(self.pool2(enc2) + ll2)
        ll3 = self.wavelet_transform(enc3)

        enc4 = self.enc4(self.pool3(enc3) + ll3)
        ll4 = self.wavelet_transform(enc4)

        enc5 = self.enc5(self.pool4(enc4) + ll4)
        ll5 = self.wavelet_transform(enc5)

        enc6 = self.enc6(self.pool5(enc5) + ll5)

        # Bottleneck
        bottleneck = self.bottleneck(self.pool5(enc6))
        print("bottleneck.shape",bottleneck.shape)
        # Decoder path
        up5 = self.up5(bottleneck)
        print("up5.shape",up5.shape)
        print("enc6.shape",enc6.shape)
        dec5 = self.dec5(torch.cat((up5, enc6), dim=1))

        up4 = self.up4(dec5)
        dec4 = self.dec4(torch.cat((up4, enc5), dim=1))

        up3 = self.up3(dec4)
        dec3 = self.dec3(torch.cat((up3, enc4), dim=1))

        up2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat((up2, enc3), dim=1))

        up1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat((up1, enc2), dim=1))

        # Final output
        out = self.final_conv(dec1)
        return out
# Example Usage
if __name__ == "__main__":
    # Example input: (batch_size, channels, depth, height, width)
    x = torch.randn(1, 1, 128, 128, 128).cuda()  # 3D medical image

    model = WaveletUNet3D(in_channels=1, out_channels=2).cuda()
    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)