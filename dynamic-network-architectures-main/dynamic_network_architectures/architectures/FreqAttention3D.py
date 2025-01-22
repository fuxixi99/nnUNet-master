import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
# from .FrequencySelection3D import *
# from .Ommiattention import *
import torch_dct as dct
class FrequencySelection3D(nn.Module):
    def __init__(self, 
                 in_channels,
                 k_list=[2],
                 lowfreq_att=True,
                 fs_feat='feat',
                 lp_type='freq',
                 act='sigmoid',
                 spatial='conv',
                 spatial_group=1,
                 spatial_kernel=3,
                 init='zero',
                 global_selection=False):
        super().__init__()
        self.k_list = k_list
        self.lp_list = nn.ModuleList()
        self.freq_weight_conv_list = nn.ModuleList()
        self.fs_feat = fs_feat
        self.lp_type = lp_type
        self.in_channels = in_channels
        if spatial_group > 64: 
            spatial_group = in_channels
        self.spatial_group = spatial_group
        self.lowfreq_att = lowfreq_att
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(in_channels // 4, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        if spatial == 'conv':
            self.freq_weight_conv_list = nn.ModuleList()
            _n = len(k_list)
            if lowfreq_att: 
                _n += 1
            for i in range(_n):
                freq_weight_conv = nn.Conv3d(
                    in_channels=in_channels, 
                    out_channels=self.spatial_group, 
                    stride=1,
                    kernel_size=spatial_kernel, 
                    groups=self.spatial_group,
                    padding=spatial_kernel // 2, 
                    bias=True
                )
                if init == 'zero':
                    # freq_weight_conv.weight.data.zero_()
                    # freq_weight_conv.bias.data.zero_()   
                    nn.init.kaiming_uniform_(freq_weight_conv.weight, mode='fan_in', nonlinearity='relu')  # He 初始化
                    freq_weight_conv.bias.data.uniform_(0.1, 0.2)  # 偏置初始化为小正值
                self.freq_weight_conv_list.append(freq_weight_conv)
        else:
            raise NotImplementedError
        
        if self.lp_type == 'avgpool':
            for k in k_list:
                self.lp_list.append(nn.Sequential(
                    nn.ReplicationPad3d(padding=k // 2),
                    nn.AvgPool3d(kernel_size=k, padding=0, stride=1)
                ))
        elif self.lp_type == 'laplacian':
            pass
        elif self.lp_type == 'freq':
            pass
        else:
            raise NotImplementedError
        
        self.act = act
        
        self.global_selection = global_selection
        if self.global_selection:
            self.global_selection_conv_real = nn.Conv3d(
                in_channels=in_channels, 
                out_channels=self.spatial_group, 
                stride=1,
                kernel_size=1, 
                groups=self.spatial_group,
                padding=0, 
                bias=True
            )
            self.global_selection_conv_imag = nn.Conv3d(
                in_channels=in_channels, 
                out_channels=self.spatial_group, 
                stride=1,
                kernel_size=1, 
                groups=self.spatial_group,
                padding=0, 
                bias=True
            )
            if init == 'zero':
                self.global_selection_conv_real.weight.data.zero_()
                self.global_selection_conv_real.bias.data.zero_()  
                self.global_selection_conv_imag.weight.data.zero_()
                self.global_selection_conv_imag.bias.data.zero_()  

    def sp_act(self, freq_weight):
        if self.act == 'sigmoid':
            # freq_weight = freq_weight.sigmoid() * 2
            freq_weight = freq_weight.sigmoid()
        elif self.act == 'softmax':
            # freq_weight = freq_weight.softmax(dim=1) * freq_weight.shape[1]
            freq_weight = freq_weight.softmax(dim=1)
        else:
            raise NotImplementedError
        return freq_weight

    def forward(self, x, att_feat=None):
        if att_feat is None: 
            att_feat = x
        x_list = []
        if self.lp_type == 'avgpool':
            pre_x = x
            b, _, d, h, w = x.shape
            for idx, avg in enumerate(self.lp_list):
                low_part = avg(x)
                high_part = pre_x - low_part
                pre_x = low_part
                freq_weight = self.freq_weight_conv_list[idx](att_feat)
                freq_weight = self.sp_act(freq_weight)
                tmp = freq_weight.reshape(b, self.spatial_group, -1, d, h, w) * high_part.reshape(b, self.spatial_group, -1, d, h, w)
                x_list.append(tmp.reshape(b, -1, d, h, w))
            if self.lowfreq_att:
                freq_weight = self.freq_weight_conv_list[len(x_list)](att_feat)
                tmp = freq_weight.reshape(b, self.spatial_group, -1, d, h, w) * pre_x.reshape(b, self.spatial_group, -1, d, h, w)
                x_list.append(tmp.reshape(b, -1, d, h, w))
            else:
                x_list.append(pre_x)
        elif self.lp_type == 'freq':
            pre_x = x.clone()
            b, _, d, h, w = x.shape
            x_fft = torch.fft.fftshift(torch.fft.fftn(x, dim=(2, 3, 4), norm='ortho'))
            if self.global_selection:
                x_real = x_fft.real
                x_imag = x_fft.imag
                global_att_real = self.global_selection_conv_real(x_real)
                global_att_real = self.sp_act(global_att_real).reshape(b, self.spatial_group, -1, d, h, w)
                global_att_imag = self.global_selection_conv_imag(x_imag)
                global_att_imag = self.sp_act(global_att_imag).reshape(b, self.spatial_group, -1, d, h, w)
                x_real = x_real.reshape(b, self.spatial_group, -1, d, h, w)
                x_imag = x_imag.reshape(b, self.spatial_group, -1, d, h, w)
                x_fft_real_updated = x_real * global_att_real
                x_fft_imag_updated = x_imag * global_att_imag
                x_fft_updated = torch.complex(x_fft_real_updated, x_fft_imag_updated)
                x_fft = x_fft_updated.reshape(b, -1, d, h, w)

            for idx, freq in enumerate(self.k_list):
                mask = torch.zeros_like(x[:, 0:1, :, :, :], device=x.device)
                mask[:, :, 
                     round(d/2 - d/(2 * freq)):round(d/2 + d/(2 * freq)), 
                     round(h/2 - h/(2 * freq)):round(h/2 + h/(2 * freq)), 
                     round(w/2 - w/(2 * freq)):round(w/2 + w/(2 * freq))] = 1.0
                low_part = torch.fft.ifftn(torch.fft.ifftshift(x_fft * mask), dim=(2, 3, 4), norm='ortho').real
                high_part = pre_x - low_part
                pre_x = low_part
                freq_weight = self.freq_weight_conv_list[idx](att_feat)
                freq_weight = self.sp_act(freq_weight)
                freq_weight = freq_weight * self.attention(att_feat)  # 使用注意力引导
                freq_weight_loss = torch.mean(freq_weight ** 2)  # 添加正则化
                # print(f"freq_weight at layer {idx}: {freq_weight.mean().item()}")
                tmp = freq_weight.reshape(b, self.spatial_group, -1, d, h, w) * high_part.reshape(b, self.spatial_group, -1, d, h, w)
                x_list.append(tmp.reshape(b, -1, d, h, w))
            if self.lowfreq_att:
                freq_weight = self.freq_weight_conv_list[len(x_list)](att_feat)
                freq_weight = self.sp_act(freq_weight)
                tmp = freq_weight.reshape(b, self.spatial_group, -1, d, h, w) * pre_x.reshape(b, self.spatial_group, -1, d, h, w)
                x_list.append(tmp.reshape(b, -1, d, h, w))
            else:
                x_list.append(pre_x)
        x = sum(x_list)
        # print()
        # output_string="freq_weight"+str(freq_weight)+"\n"
        # with open(f"freq_test.txt", "a") as file:
        #     file.write(output_string)
        output_string = f"freq_weight mean: {freq_weight.mean().item()}\n"
        with open("freq_test11.txt", "a") as file:
            file.write(output_string)

        return x

# class OmniAttention(nn.Module):
#     """
#     For adaptive kernel, AdaKern (3D version)
#     """
#     def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
#         super(OmniAttention, self).__init__()
#         attention_channel = max(int(in_planes * reduction), min_channel)
#         self.kernel_size = kernel_size
#         self.kernel_num = kernel_num
#         self.temperature = 1.0

#         self.avgpool = nn.AdaptiveAvgPool3d(1)
#         self.fc = nn.Conv3d(in_planes, attention_channel, 1, bias=False)
#         self.bn = nn.BatchNorm3d(attention_channel)
#         self.relu = nn.ReLU(inplace=True)

#         self.channel_fc = nn.Conv3d(attention_channel, in_planes, 1, bias=True)
#         self.func_channel = self.get_channel_attention

#         if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
#             self.func_filter = self.skip
#         else:
#             self.filter_fc = nn.Conv3d(attention_channel, out_planes, 1, bias=True)
#             self.func_filter = self.get_filter_attention

#         if kernel_size == 1:  # point-wise convolution
#             self.func_spatial = self.skip
#         else:
#             self.spatial_fc = nn.Conv3d(attention_channel, kernel_size * kernel_size * kernel_size, 1, bias=True)
#             self.func_spatial = self.get_spatial_attention

#         if kernel_num == 1:
#             self.func_kernel = self.skip
#         else:
#             self.kernel_fc = nn.Conv3d(attention_channel, kernel_num, 1, bias=True)
#             self.func_kernel = self.get_kernel_attention

#         self._initialize_weights()

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             if isinstance(m, nn.BatchNorm3d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def update_temperature(self, temperature):
#         self.temperature = temperature

#     @staticmethod
#     def skip(_):
#         return 1.0

#     def get_channel_attention(self, x):
#         channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1, 1) / self.temperature)
#         return channel_attention

#     def get_filter_attention(self, x):
#         filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1, 1) / self.temperature)
#         return filter_attention

#     def get_spatial_attention(self, x):
#         spatial_attention = self.spatial_fc(x).view(
#             x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size, self.kernel_size
#         )
#         spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
#         return spatial_attention

#     def get_kernel_attention(self, x):
#         kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1, 1)
#         kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
#         return kernel_attention

#     def forward(self, x):
#         x = self.avgpool(x)
#         x = self.fc(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)

class OmniAttention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(OmniAttention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Conv3d(in_planes, attention_channel, 1, bias=False)
        # self.bn = nn.BatchNorm3d(attention_channel)
        self.bn = nn.GroupNorm(num_groups=8, num_channels=24)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv3d(attention_channel, in_planes, 1, bias=True)
        self.filter_fc = nn.Conv3d(attention_channel, out_planes, 1, bias=True)
        self.spatial_fc = nn.Conv3d(attention_channel, kernel_size**3, 1, bias=True)
        self.kernel_fc = nn.Conv3d(attention_channel, kernel_num, 1, bias=True)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)

        channel_attention = torch.sigmoid(self.channel_fc(x))
        filter_attention = torch.sigmoid(self.filter_fc(x))
        spatial_attention = torch.sigmoid(self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size, self.kernel_size))
        kernel_attention = F.softmax(self.kernel_fc(x).view(x.size(0), self.kernel_num, 1, 1, 1, 1, 1), dim=1)

        return channel_attention, filter_attention, spatial_attention, kernel_attention



class FreqAttention3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, kernel_num=4, groups=1):
        super(FreqAttention3D, self).__init__()
        self.freq_select = FrequencySelection3D(in_channels=in_channels)
        self.omni_attention = OmniAttention(
            in_planes=in_channels,
            out_planes=out_channels,
            kernel_size=kernel_size,
            kernel_num=kernel_num
        )
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.adaptive_weight_mean = nn.Parameter(torch.randn(2, 384, 1, 1, 1))
        self.adaptive_weight_res = nn.Parameter(torch.randn(2, 384, 1, 1, 1))
    def forward(self, x):
        # Step 1: Frequency Select
        freq_features = self.freq_select(x)
        
        # Step 2: OmniAttention
        # c_att1, f_att1, _, _ = self.omni_attention(freq_features)
        # c_att2, f_att2, _, _ = self.omni_attention(freq_features)

        # # print("c_att1 shape:", c_att1.shape)
        # # print("f_att1 shape:", f_att1.shape)
        # # print("c_att2 shape:", c_att2.shape)
        # # print("f_att2 shape:", f_att2.shape)

        # print("adaptive_weight_mean shape before reshape:", self.adaptive_weight_mean.shape)
        # print("adaptive_weight_res shape before reshape:", self.adaptive_weight_res.shape)
        # b = freq_features.shape[0]
        # adaptive_weight = self.adaptive_weight_mean.unsqueeze(0).repeat(b, 1, 1, 1, 1,1)  # b, c_out, c_in, k, k
        # adaptive_weight_mean = adaptive_weight.mean(dim=(-1, -2), keepdim=True)
        # adaptive_weight_res = adaptive_weight - adaptive_weight_mean

        # # 你可以在这里调整形状或者进行其他计算，假设 c_att1 和 f_att1 的形状是 [b, 384, 1, 1, 1]
        # _, c_out, c_in, k, k,k = adaptive_weight.shape

        # # 如果需要使用 DCT（离散余弦变换），你可以继续使用类似的方式：
        # # if self.use_dct:
        # #     dct_coefficients = dct.dct_3d(adaptive_weight_res)
            
        # #     # 假设 spatial_att2 是需要的空间注意力项，你可以调整它的形状以匹配
        # #     spatial_att2 = spatial_att2.reshape(b, 1, 1, k, k)
        # #     dct_coefficients = dct_coefficients * (spatial_att2 * 2)  # 加权 DCT 系数
            
        # #     adaptive_weight_res = dct.idct_3d(dct_coefficients)

        # # 这里的 adaptive_weight 是你需要的权重，包含了权重的均值和调整后的残差项
        # adaptive_weight = adaptive_weight_mean * (c_att1.unsqueeze(1) * 2) * (f_att1.unsqueeze(2) * 2) + adaptive_weight_res * (c_att2.unsqueeze(1) * 2) * (f_att2.unsqueeze(2) * 2)

        # # 最后，你可以 reshape 结果，以便于下一步操作
        # adaptive_weight = adaptive_weight.reshape(-1, self.in_channels // self.groups, 3, 3)  # 调整为需要的形状

        # if self.bias is not None:
        #     adaptive_weight = adaptive_weight + self.bias

        # # Step 4: Apply adaptive weight
        # output = F.conv3d(
        #     freq_features,
        #     adaptive_weight,
        #     stride=1,
        #     padding=self.kernel_size // 2,
        #     groups=self.groups
        # )
        # OmniAttention
        channel_att, filter_att, spatial_att, kernel_att = self.omni_attention(freq_features)
        # print("channel_att shape:", channel_att.shape)
        # print("filter_att shape:", filter_att.shape)
        # print("spatial_att shape:", spatial_att.shape)
        # print("kernel_att shape:", kernel_att.shape)
        # Feature Fusion
        enhanced_features = freq_features * channel_att + filter_att
        return enhanced_features
if __name__ == '__main__':
    x = torch.rand(2, 384, 8,8, 8).cuda()
    # m = AdaptiveDilatedConv(in_channels=4, out_channels=8, kernel_size=3).cuda()
    m=FreqAttention3D(in_channels=384,out_channels=384).cuda()
    m.eval()
    y = m(x)
    print(y.shape)